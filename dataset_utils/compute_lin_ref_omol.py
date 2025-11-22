from __future__ import annotations

"""
Compute linear reference coefficients for omol 4million dataset.
python compute_lin_ref_4mil.py --dataset_path /checkpoint/ocp/manasakani/omol_energies_and_forces/train_4M --num-workers 4 --stats-dir stats_omol_4mil --dataset-type omol --subsample 0.001
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
from fairchem.core.datasets import AseDBDataset


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
        default=os.cpu_count(),
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

    # Add this to your parse_args() function
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["nablaDFT", "omol"],
        default="nablaDFT",
        help="Type of dataset to process"
    )

    return parser.parse_args()

def extract_stats(idx):
    dataset = AseDBDataset({"src": dataset_path})

    atoms = dataset.get_atoms(idx)
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(atomic_numbers)
    energy = atoms.get_potential_energy()
    # forces = atoms.get_forces()

    # fixed_idx = np.zeros(n_atoms)
    # if hasattr(atoms, "constraints"):
    #     from ase.constraints import FixAtoms

    #     for constraint in atoms.constraints:
    #         if isinstance(constraint, FixAtoms):
    #             fixed_idx[constraint.index] = 1

    # mask = fixed_idx == 0
    # forces = forces[mask]
    lin_energy = sum(lin_ref[atomic_numbers])
    energy -= lin_energy

    return (energy, n_atoms)

def compute_stats(num_workers, stats_out_path, indices):
    pool = Pool(num_workers)
    outputs = list(tqdm(pool.imap(extract_stats, indices), total=len(indices)))

    energies = [x[0] for x in outputs]
    # forces = np.array([force for x in outputs for force in x[1]])
    num_atoms = [x[1] for x in outputs]

    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    # force_rms = np.sqrt(np.mean(np.square(forces)))
    # force_norms = np.linalg.norm(forces, axis=-1)
    # force_md = np.mean(force_norms)
    avg_num_atoms = np.mean(num_atoms)

    print(
        f"energy_mean: {energy_mean}, energy_std: {energy_std}, avg_num_atoms: {avg_num_atoms}"
    )
    # write stats to file
    stats_file = os.path.join(stats_out_path, "stats.pkl")
    with open(stats_file, "wb") as f:
        pickle.dump(
            {
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                # "force_rms": force_rms,
                # "force_md": force_md,
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
    # force_norm_hist, force_norm_bins = np.histogram(force_norms, bins=100)
    # fhist_file = os.path.join(stats_out_path, "force_norm_histogram.npz")
    # np.savez(fhist_file, hist=force_norm_hist, bins=force_norm_bins)


def extract_data(idx):

    # print(f"Extracting data for index {idx}")
    dataset = AseDBDataset({"src": dataset_path})
    atoms = dataset.get_atoms(idx)
    atomic_numbers = atoms.get_atomic_numbers()

    # Count atoms by type for linear regression features
    # print("Counting bins...")
    x = np.bincount(atomic_numbers, minlength=max_atom_types).astype(int)
    y = atoms.get_potential_energy() # in eV

    return (x, y)

def compute_lin_ref(num_workers, indices, stats_dir):

    if num_workers == 1:
        outputs = [extract_data(idx) for idx in tqdm(indices, total=len(indices))]
    else:
        pool = Pool(num_workers)
        outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))
    print("Done extracting data.")

    print("getting linear problem")
    features = [x[0] for x in outputs]
    targets = [x[1] for x in outputs]

    print("stacking features")
    X = np.vstack(features)
    y = targets

    print("solving lstsq problem")
    coeff = np.linalg.lstsq(X, y, rcond=None)[0]

    print("saving results")
    # we'd use this at eval time to predict hof values
    np.savez_compressed(
        os.path.join(stats_dir, "lin_ref_coeffs_hof.npz"),
        coeff=coeff,
    )
    # we'd use this for training rather than modifying the dataset
    np.savez_compressed(
        os.path.join(stats_dir, "joint_hof_lin_coeffs.npz"),
        coeff=coeff #+ hof_reference,
    )

def main_omol(args):
    """Main function for OMOL dataset processing"""
    stats_dir = os.path.join(args.stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    print("stats dir is", stats_dir)

    global max_atom_types, dataset_path
    max_atom_types = args.max_atom_types
    dataset_path = args.dataset_path

    # Construct database paths
    dataset = AseDBDataset({"src": dataset_path})
    total_molecules_all = len(dataset)

    if args.subsample is not None:
        args.subsample = int(min(len(dataset), args.subsample * len(dataset)))
        indices = torch.randperm(len(dataset))[: args.subsample].tolist()
    else:
        indices = range(len(dataset))

    print(f"Computing linear reference coefficients for {len(indices)} OMOL molecules...")

    compute_lin_ref(args.num_workers, indices, stats_dir)
    print("done extracting refs")
    # load lin refs
    global lin_ref
    lin_ref_out_file = os.path.join(args.stats_dir, "joint_hof_lin_coeffs.npz")
    lin_ref = np.load(lin_ref_out_file, allow_pickle=True)["coeff"]

    print("Computing stats")
    compute_stats(args.num_workers, stats_dir, indices)

if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    # Parse arguments FIRST
    args = parse_args()

    main_omol(args)

    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
