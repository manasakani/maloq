from __future__ import annotations

"""
Compute linear reference coefficients for nablaDFT dataset.

Example usage:
python compute_lin_ref.py --dataset_path ./fock_datasets/nabla2_DFT/train_10k.db --num-workers 4 --stats-dir stats_nablaDFT
"""

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from dataset_utils.nablaDFT_dataset_utils import HamiltonianDatabase
from ase import Atoms


def extract_data(idx):

    print(f"Extracting data for index {idx}")
    
    # Extract data from nablaDFT dataset
    atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = dataset[idx]
    
    # Create ASE atoms object
    atoms = Atoms(symbols=atomic_numbers, positions=positions)
    atoms.calc = None  # Remove calculator to avoid issues
    
    # Set energy manually since nablaDFT stores it separately
    atoms.info['energy'] = energy
    
    # Count atoms by type for linear regression features
    x = np.bincount(atomic_numbers, minlength=max_atom_types).astype(int)
    y = energy  # No hof_ref_energy subtraction as requested

    return (x, y)


def compute_lin_ref(num_workers, indices, stats_dir):
    pool = Pool(num_workers)
    outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))

    features = [x[0] for x in outputs]
    targets = [x[1] for x in outputs]

    X = np.vstack(features)
    y = targets

    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    # Save linear reference coefficients for nablaDFT dataset
    np.savez_compressed(
        os.path.join(stats_dir, "lin_ref_coeffs_nablaDFT.npz"),
        coeff=coeff,
    )
    # Save coefficients for training use
    np.savez_compressed(
        os.path.join(stats_dir, "joint_nablaDFT_lin_coeffs.npz"),
        coeff=coeff,
    )


def extract_stats(idx):
    # Extract data from nablaDFT dataset
    atomic_numbers, positions, energy, forces, hamiltonian, overlap, coeff_matrix, moses_id, conformation_id = dataset[idx]
    
    # Create ASE atoms object
    atoms = Atoms(symbols=atomic_numbers, positions=positions)
    n_atoms = len(atomic_numbers)
    
    # Use nablaDFT energy and forces directly
    # No fixed atoms constraints in nablaDFT dataset, so use all forces
    lin_energy = sum(lin_ref[atomic_numbers])
    energy -= lin_energy

    return (energy, forces, n_atoms)


def compute_stats(num_workers, stats_out_path, indices):
    pool = Pool(num_workers)

    extract_stats(indices[0])
    outputs = list(tqdm(pool.imap(extract_stats, indices), total=len(indices)))

    energies = [x[0] for x in outputs]
    forces = np.array([force for x in outputs for force in x[1]])
    num_atoms = [x[2] for x in outputs]

    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    force_rms = np.sqrt(np.mean(np.square(forces)))
    force_norms = np.linalg.norm(forces, axis=-1)
    force_md = np.mean(force_norms)
    avg_num_atoms = np.mean(num_atoms)

    print(
        f"energy_mean: {energy_mean}, energy_std: {energy_std}, force_rms: {force_rms}, force_md: {force_md}, avg_num_atoms: {avg_num_atoms}"
    )
    # write stats to file
    stats_file = os.path.join(stats_out_path, "stats.pkl")
    with open(stats_file, "wb") as f:
        pickle.dump(
            {
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "force_rms": force_rms,
                "force_md": force_md,
                "avg_num_atoms": avg_num_atoms,
            },
            f,
        )

    # Compute and save energy histogram
    energy_hist, energy_bins = np.histogram(energies, bins=100)
    ehist_file = os.path.join(stats_out_path, "energy_histogram.npz")
    np.savez(ehist_file, hist=energy_hist, bins=energy_bins)
    # Compute force norms
    # Compute and save force norm histogram
    force_norm_hist, force_norm_bins = np.histogram(force_norms, bins=100)
    fhist_file = os.path.join(stats_out_path, "force_norm_histogram.npz")
    np.savez(fhist_file, hist=force_norm_hist, bins=force_norm_bins)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-atom-types",
        type=int,
        default=100,
        help="maximum number of atom types in the dataset",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="path to nablaDFT HamiltonianDatabase (.db file)",
    )
    parser.add_argument(
        "--hof-reference",
        help="path to hof reference (not used for nablaDFT)",
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="number of workers to use for processing files",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="subsample the dataset of lin refs, in (0,1]",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="stats",
        help="subdir for stats",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats_dir = os.path.join(args.stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    print("stats dir is", stats_dir)

    global dataset
    # Load nablaDFT dataset
    dataset = HamiltonianDatabase(args.dataset_path)
    print(f"Loaded nablaDFT dataset with {len(dataset)} molecules")
    
    global max_atom_types
    max_atom_types = args.max_atom_types

    # HOF reference not used for nablaDFT as per user request
    global hof_reference
    hof_reference = None

    if args.subsample is not None:
        args.subsample = int(min(len(dataset), args.subsample * len(dataset)))
        indices = torch.randperm(len(dataset))[: args.subsample].tolist()
    else:
        indices = range(len(dataset))
    
    print(f"Computing linear reference coefficients for {len(indices)} molecules...")
    compute_lin_ref(args.num_workers, indices, stats_dir)
    
    # Load computed coefficients for stats computation
    global lin_ref
    lin_ref_out_file = os.path.join(args.stats_dir, "joint_nablaDFT_lin_coeffs.npz")
    lin_ref = np.load(lin_ref_out_file, allow_pickle=True)["coeff"]
    
    print("Computing dataset statistics...")
    compute_stats(args.num_workers, stats_dir, indices)


if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    # Call the main function
    main()
    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")