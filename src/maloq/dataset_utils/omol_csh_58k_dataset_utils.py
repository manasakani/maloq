# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
import os
import h5py
import numpy as np
import torch
import json
from torch.utils.data import Dataset

from ..fock_utils import utils_orca_out, basis_sets

def restore_orca_diffuse_order(h_matrix, atomic_numbers, correct_basis_dict=None):
    """Undoes the invalid `orca_to_e3nn` transformation that used an L-sorted basis

    (from ORCA output counts), restoring raw ORCA order, and re-applies
    `orca_to_e3nn` using the correct `def2_tzvpd` shell order.

    Parameters
    ----------
    h_matrix : np.ndarray or torch.Tensor
        2D matrix (N x N) or 1D flat upper-triangular vector.
    atomic_numbers : list or np.ndarray
        List of atomic numbers Z for the molecule/system.
    correct_basis_dict : dict, optional
        Dict mapping Z (int) -> list of shell l-values (defaults to def2_tzvpd).

    Returns
    -------
    np.ndarray or torch.Tensor
        Corrected matrix in proper e3nn order.
    """
    if correct_basis_dict is None:
        correct_basis_dict = {
            utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element]
            for element in basis_sets.def2_tzvpd.keys()
        }
        correct_basis_dict = {int(k): v for k, v in correct_basis_dict.items()}

    # 1. Reconstruct the L-sorted dynamic basis used during dataset creation
    dynamic_basis_dict = {
        int(z): sorted(correct_basis_dict[int(z)])
        for z in set(atomic_numbers)
    }

    # Handle torch tensor vs numpy array
    is_torch = isinstance(h_matrix, torch.Tensor)
    device = h_matrix.device if is_torch else None
    dtype = h_matrix.dtype if is_torch else None
    mat_np = h_matrix.detach().cpu().numpy() if is_torch else np.array(h_matrix)

    # 2. Re-inflate if flat 1D upper-triangular array
    if mat_np.ndim == 1:
        mat_np = reinflate_symmetric_matrix(mat_np)

    # 3. Step 1: Undo the corrupted transformation back to raw ORCA order
    h_raw_orca = utils_orca_out.sort_by_m(
        mat_np, dynamic_basis_dict, atomic_numbers, direction="e3nn_to_orca"
    )

    # 4. Step 2: Redo orca_to_e3nn using the correct def2_tzvpd shell order
    h_correct_e3nn = utils_orca_out.sort_by_m(
        h_raw_orca, correct_basis_dict, atomic_numbers, direction="orca_to_e3nn"
    )

    if is_torch:
        return torch.from_numpy(h_correct_e3nn).to(device=device, dtype=dtype)
    return h_correct_e3nn

class OMol_CSH_58k_Database(Dataset):
    """
    PyTorch Dataset wrapper for the OMol CSH 58k (and related 1k test) HDF5 datasets.
    Optimized for zero-copy memory transfers, fast distributed subsetting via JSON manifest,
    and high-throughput chunk caching.
    """
    def __init__(self, dbpath: str, indices: list = None):
        super().__init__()
        if not os.path.exists(dbpath):
            raise FileNotFoundError(f"HDF5 database not found at path: {dbpath}")

        self.dbpath = dbpath
        self._file = None 

        # Load keys from the pre-computed JSON manifest
        manifest_path = f"{self.dbpath}.keys.json"
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Key manifest not found at {manifest_path}. "
                "Please run the 'save_keys.py' script first to generate it."
            )

        with open(manifest_path, 'r') as f:
            all_keys = json.load(f)

        # Subset the keys if the DataLoader requested a specific rank slice
        if indices is not None:
            self.molecule_keys = [all_keys[i] for i in indices]
            print(f"Dataset initialized: {len(self.molecule_keys)} molecules loaded for this rank subset.", flush=True)
        else:
            self.molecule_keys = all_keys
            print(f"Dataset initialized: {len(self.molecule_keys)} global molecules loaded.", flush=True)

    def _get_file(self) -> h5py.File:
        """Lazy per-process open with expanded chunk cache (64MB) and slot tuning."""
        if self._file is None:
            self._file = h5py.File(
                self.dbpath, 
                'r', 
                rdcc_nbytes=64 * 1024 * 1024,  # 64 MB chunk cache (default is 1 MB)
                swmr=True                      # Single-writer-multiple-reader mode
            )
        return self._file

    @classmethod
    def _reinflate_symmetric_matrix(cls, flat_array: np.ndarray) -> np.ndarray:
        n = int((np.sqrt(8 * len(flat_array) + 1) - 1) // 2)
        
        mat = np.zeros((n, n), dtype=flat_array.dtype)
        mat[np.triu_indices(n)] = flat_array
        mat = mat + mat.T - np.diag(mat.diagonal())
        
        return mat

    def __len__(self) -> int:
        return len(self.molecule_keys)

    def __getitem__(self, idx: int) -> dict:
        f = self._get_file()
        key = self.molecule_keys[idx]
        grp = f[key]

        fock_flat = grp['fock'][:]
        elements = grp['elements'][:]
        coords = grp['coords'][:]

        fock = self._reinflate_symmetric_matrix(fock_flat)

        # For OMol_CSH_58k, correct the Fock matrix ordering if it was permuted incorrectly by ORCA's output
        fock = restore_orca_diffuse_order(fock, elements)

        charge = float(grp.attrs.get('charge', 0))
        spin = float(grp.attrs.get('spin', 0))

        sample = {
            'z': torch.from_numpy(elements).to(torch.long),
            'pos': torch.from_numpy(coords).to(torch.float32),
            'fock': torch.from_numpy(fock).to(torch.float32),
            'charge': torch.tensor(charge, dtype=torch.long),
            'spin': torch.tensor(spin, dtype=torch.long),
            'name': key
        }

        return sample

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()