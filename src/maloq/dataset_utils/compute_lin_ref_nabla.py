from __future__ import annotations

"""
Compute linear reference coefficients for nablaDFT dataset.

Example usage:
python compute_lin_ref_nabla.py --dataset_path /capstor/store/cscs/pasc/c33/manasa/nablaDFT_datasets/train_10k.db --num-workers 4 --stats-dir stats_nablaDFT
"""

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from .nablaDFT_dataset_utils import HamiltonianDatabase
from ase import Atoms


def extract_data(idx):
    
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

    # Add this to your parse_args() function
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["nablaDFT", "omol"],
        default="nablaDFT",
        help="Type of dataset to process"
    )

    return parser.parse_args()

def extract_data_omol(idx):
    """Extract data for OMOL dataset"""
    print(f"Extracting OMOL data for index {idx}")
    
    # Get structure by id from ASE database
    structure = dataset.get(dataset_ids[idx])
    
    # Extract atomic numbers and energy
    atoms = structure.toatoms()
    atomic_numbers = atoms.numbers
    
    # Extract energy from structure data
    energy = structure.data['total_energy [Eh]']  # Energy in Hartree
    
    # Count atoms by type for linear regression features
    x = np.bincount(atomic_numbers, minlength=max_atom_types).astype(int)
    y = energy
    
    return (x, y)


def extract_stats_omol(idx):
    """Extract stats for OMOL dataset"""
    # Get structure by id from ASE database
    structure = dataset.get(dataset_ids[idx])
    
    # Extract atomic numbers, energy, and forces
    atoms = structure.toatoms()
    atomic_numbers = atoms.numbers
    n_atoms = len(atomic_numbers)
    
    # Extract energy and forces from structure data
    energy = structure.data['total_energy [Eh]']  # Energy in Hartree
    forces = structure.data['gradient [Eh/bohr]']  # Forces in Hartree/bohr
    
    # Apply linear reference correction
    lin_energy = sum(lin_ref[atomic_numbers])
    energy -= lin_energy
    
    return (energy, forces, n_atoms)


def compute_lin_ref_omol(num_workers, indices, stats_dir, db_path):
    """Compute linear reference for OMOL dataset"""
    global dataset, dataset_ids
    
    # Connect to OMOL ASE database
    import ase.db
    dataset = ase.db.connect(db_path)
    print(f"Connected to OMOL database: {db_path}")
    
    # Get all structure IDs
    dataset_ids = []
    for row in dataset.select():
        dataset_ids.append(row.id)
    
    print(f"Loaded OMOL dataset with {len(dataset_ids)} molecules")
    
    # Use subset of indices if provided
    if len(indices) < len(dataset_ids):
        dataset_ids = [dataset_ids[i] for i in indices]
    
    pool = Pool(num_workers)
    outputs = list(tqdm(pool.imap(extract_data_omol, range(len(dataset_ids))), total=len(dataset_ids)))
    
    features = [x[0] for x in outputs]
    targets = [x[1] for x in outputs]
    
    X = np.vstack(features)
    y = targets
    
    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Save linear reference coefficients for OMOL dataset
    np.savez_compressed(
        os.path.join(stats_dir, "lin_ref_coeffs_omol.npz"),
        coeff=coeff,
    )
    # Save coefficients for training use
    np.savez_compressed(
        os.path.join(stats_dir, "joint_omol_lin_coeffs.npz"),
        coeff=coeff,
    )
    
    return coeff


def compute_stats_omol(num_workers, stats_out_path, indices, db_path):
    """Compute stats for OMOL dataset after linear reference correction"""
    global dataset, dataset_ids, lin_ref
    
    # Connect to OMOL ASE database
    import ase.db
    dataset = ase.db.connect(db_path)
    
    # Get all structure IDs
    dataset_ids = []
    for row in dataset.select():
        dataset_ids.append(row.id)
    
    # Use subset of indices if provided
    if len(indices) < len(dataset_ids):
        dataset_ids = [dataset_ids[i] for i in indices]
    
    pool = Pool(num_workers)
    
    # Test with first sample
    extract_stats_omol(0)
    outputs = list(tqdm(pool.imap(extract_stats_omol, range(len(dataset_ids))), total=len(dataset_ids)))
    
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
        f"OMOL - energy_mean: {energy_mean}, energy_std: {energy_std}, force_rms: {force_rms}, force_md: {force_md}, avg_num_atoms: {avg_num_atoms}"
    )
    
    # Write stats to file
    stats_file = os.path.join(stats_out_path, "stats_omol.pkl")
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
    ehist_file = os.path.join(stats_out_path, "energy_histogram_omol.npz")
    np.savez(ehist_file, hist=energy_hist, bins=energy_bins)
    
    # Compute and save force norm histogram
    force_norm_hist, force_norm_bins = np.histogram(force_norms, bins=100)
    fhist_file = os.path.join(stats_out_path, "force_norm_histogram_omol.npz")
    np.savez(fhist_file, hist=force_norm_hist, bins=force_norm_bins)

def compute_lin_ref_omol_multi(num_workers, db_paths, total_subsample, stats_dir):
    """Compute linear reference for multiple OMOL database files"""
    
    all_features = []
    all_targets = []
    molecules_processed = 0
    
    for db_path in db_paths:
        print(f"Processing database: {db_path}")
        
        # Connect to current database
        import ase.db
        dataset = ase.db.connect(db_path)
        
        # Get all structure IDs for this database
        dataset_ids = []
        for row in dataset.select():
            dataset_ids.append(row.id)
        
        # Determine how many molecules to process from this database
        if total_subsample < float('inf'):
            # Proportional subsampling
            db_molecule_count = len(dataset_ids)
            remaining_to_process = total_subsample - molecules_processed
            molecules_from_this_db = min(db_molecule_count, remaining_to_process)
            
            if molecules_from_this_db <= 0:
                break
                
            # Randomly sample from this database
            import random
            dataset_ids = random.sample(dataset_ids, molecules_from_this_db)
        
        print(f"Processing {len(dataset_ids)} molecules from {db_path}")
        
        # Process molecules from current database
        features_db = []
        targets_db = []
        
        for idx in tqdm(range(len(dataset_ids)), desc=f"Processing {os.path.basename(db_path)}"):
            # Get structure by id from ASE database
            structure = dataset.get(dataset_ids[idx])
            
            # Extract atomic numbers and energy
            atoms = structure.toatoms()
            atomic_numbers = atoms.numbers
            
            # Extract energy from structure data
            energy = structure.data['total_energy [Eh]']  # Energy in Hartree
            
            # Count atoms by type for linear regression features
            x = np.bincount(atomic_numbers, minlength=max_atom_types).astype(int)
            y = energy
            
            features_db.append(x)
            targets_db.append(y)
        
        all_features.extend(features_db)
        all_targets.extend(targets_db)
        molecules_processed += len(dataset_ids)
        
        print(f"Processed {molecules_processed} molecules so far")
        
        # Stop if we've reached our target
        if molecules_processed >= total_subsample:
            break
    
    print(f"Total molecules processed: {len(all_features)}")
    
    # Combine all features and targets
    X = np.vstack(all_features)
    y = np.array(all_targets)
    
    # Compute linear regression coefficients
    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Save linear reference coefficients for OMOL dataset
    np.savez_compressed(
        os.path.join(stats_dir, "lin_ref_coeffs_omol.npz"),
        coeff=coeff,
    )
    # Save coefficients for training use
    np.savez_compressed(
        os.path.join(stats_dir, "joint_omol_lin_coeffs.npz"),
        coeff=coeff,
    )
    
    print(f"Saved linear reference coefficients based on {len(all_features)} molecules")
    return coeff


def compute_stats_omol_multi(num_workers, stats_out_path, db_paths, total_subsample):
    """Compute stats for multiple OMOL database files after linear reference correction"""
    
    all_energies = []
    all_forces = []
    all_num_atoms = []
    molecules_processed = 0
    
    for db_path in db_paths:
        print(f"Computing stats for database: {db_path}")
        
        # Connect to current database
        import ase.db
        dataset = ase.db.connect(db_path)
        
        # Get all structure IDs for this database
        dataset_ids = []
        for row in dataset.select():
            dataset_ids.append(row.id)
        
        # Determine how many molecules to process from this database
        if total_subsample < float('inf'):
            # Proportional subsampling
            db_molecule_count = len(dataset_ids)
            remaining_to_process = total_subsample - molecules_processed
            molecules_from_this_db = min(db_molecule_count, remaining_to_process)
            
            if molecules_from_this_db <= 0:
                break
                
            # Use same random seed as before for consistency
            import random
            random.seed(42)  # For reproducibility
            dataset_ids = random.sample(dataset_ids, molecules_from_this_db)
        
        # Process molecules from current database
        for idx in tqdm(range(len(dataset_ids)), desc=f"Stats for {os.path.basename(db_path)}"):
            # Get structure by id from ASE database
            structure = dataset.get(dataset_ids[idx])
            
            # Extract atomic numbers, energy, and forces
            atoms = structure.toatoms()
            atomic_numbers = atoms.numbers
            n_atoms = len(atomic_numbers)
            
            # Extract energy and forces from structure data
            energy = structure.data['total_energy [Eh]']  # Energy in Hartree
            forces = structure.data['gradient [Eh/bohr]']  # Forces in Hartree/bohr
            
            # Apply linear reference correction
            lin_energy = sum(lin_ref[atomic_numbers])
            energy -= lin_energy
            
            all_energies.append(energy)
            all_forces.extend(forces)
            all_num_atoms.append(n_atoms)
        
        molecules_processed += len(dataset_ids)
        
        # Stop if we've reached our target
        if molecules_processed >= total_subsample:
            break
    
    # Compute statistics
    forces_array = np.array(all_forces)
    
    energy_mean = np.mean(all_energies)
    energy_std = np.std(all_energies)
    force_rms = np.sqrt(np.mean(np.square(forces_array)))
    force_norms = np.linalg.norm(forces_array, axis=-1)
    force_md = np.mean(force_norms)
    avg_num_atoms = np.mean(all_num_atoms)
    
    print(
        f"OMOL Multi-DB - energy_mean: {energy_mean}, energy_std: {energy_std}, force_rms: {force_rms}, force_md: {force_md}, avg_num_atoms: {avg_num_atoms}"
    )
    
    # Write stats to file
    stats_file = os.path.join(stats_out_path, "stats_omol.pkl")
    with open(stats_file, "wb") as f:
        pickle.dump(
            {
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "force_rms": force_rms,
                "force_md": force_md,
                "avg_num_atoms": avg_num_atoms,
                "total_molecules": len(all_energies),
                "databases_processed": len(db_paths),
            },
            f,
        )
    
    # Compute and save energy histogram
    energy_hist, energy_bins = np.histogram(all_energies, bins=100)
    ehist_file = os.path.join(stats_out_path, "energy_histogram_omol.npz")
    np.savez(ehist_file, hist=energy_hist, bins=energy_bins)
    
    # Compute and save force norm histogram
    force_norm_hist, force_norm_bins = np.histogram(force_norms, bins=100)
    fhist_file = os.path.join(stats_out_path, "force_norm_histogram_omol.npz")
    np.savez(fhist_file, hist=force_norm_hist, bins=force_norm_bins)
    
    print(f"Computed statistics from {len(all_energies)} total molecules across {len(db_paths)} databases")

def main_omol(args):
    """Main function for OMOL dataset processing"""
    stats_dir = os.path.join(args.stats_dir)
    os.makedirs(stats_dir, exist_ok=True)
    print("stats dir is", stats_dir)
    
    global max_atom_types
    max_atom_types = args.max_atom_types
    
    # Construct all database paths
    base_path = args.dataset_path
    # Remove the job number from the path if it exists
    if 'job_' in base_path:
        base_path = base_path.rsplit('_job_', 1)[0]
    
    db_paths = []
    total_molecules_all = 0
    
    # Check which database files exist
    for job_id in range(64):
        db_path = f"{base_path}_job_{job_id}.db"
        if os.path.exists(db_path):
            import ase.db
            temp_db = ase.db.connect(db_path)
            mol_count = temp_db.count()
            if mol_count > 0:
                db_paths.append(db_path)
                total_molecules_all += mol_count
                print(f"Found database {db_path} with {mol_count} molecules")
            # Remove temp_db.close() - ASE databases don't have this method
        else:
            print(f"Database {db_path} not found, skipping...")
    
    print(f"Total molecules across all databases: {total_molecules_all}")
    print(f"Will process {len(db_paths)} database files")
    
    # Determine indices for subsampling
    if args.subsample is not None:
        subsample_count = int(min(total_molecules_all, args.subsample * total_molecules_all))
        print(f"Subsampling to {subsample_count} molecules")
    else:
        subsample_count = total_molecules_all
    
    print(f"Computing linear reference coefficients for {subsample_count} OMOL molecules...")
    coeff = compute_lin_ref_omol_multi(args.num_workers, db_paths, subsample_count, stats_dir)
    
    # Load computed coefficients for stats computation
    global lin_ref
    lin_ref_out_file = os.path.join(args.stats_dir, "joint_omol_lin_coeffs.npz")
    lin_ref = np.load(lin_ref_out_file, allow_pickle=True)["coeff"]
    
    print("Computing OMOL dataset statistics...")
    compute_stats_omol_multi(args.num_workers, stats_dir, db_paths, subsample_count)

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

    # Parse arguments FIRST
    args = parse_args()

    if args.dataset_type == "omol":
        main_omol(args)
    else:
        main()  # Original nablaDFT processing

    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")