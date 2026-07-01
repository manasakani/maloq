# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.

import argparse
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
import math

from ase import Atoms
from ase.db import connect
from ..fock_utils import utils_orca_out, utils_tensor_decomp, fock_targets, basis_sets

def parse_args():
    parser = argparse.ArgumentParser(description='Create OMOL dataset (array version)')
    parser.add_argument('-f', '--structures_dir', type=str, required=True,
                       help='Directory containing structure folders')
    parser.add_argument('-o', '--output_db_name', type=str, required=True,
                       help='Output database name (without .db extension)')
    parser.add_argument('--output_folder', type=str, default='new_database',
                          help='Folder to save the output database files')
    parser.add_argument('--start_idx', type=int, required=True,
                       help='Start index for structure processing')
    parser.add_argument('--end_idx', type=int, required=True,
                       help='End index for structure processing')
    parser.add_argument('--job_id', type=int, required=True,
                       help='SLURM array job ID')
    return parser.parse_args()

def get_subdirs(parent_dir, n):
    subdirs = []
    with os.scandir(parent_dir) as it:
        for entry in it:
            if entry.is_dir():
                subdirs.append(entry.name)
                if len(subdirs) >= n:
                    break
    return subdirs

def check_elements(elements, basis_set):
    for element in elements:
        if element not in basis_set:
            return False
    return True

def expand_density_mat(P):
    n = int((np.sqrt(8 * len(P) + 1) - 1) // 2)
    mat = np.zeros((n,n))
    mat[np.triu_indices(n)] = P
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat

def main():
    args = parse_args()

    print(f"Job {args.job_id}: Processing structures {args.start_idx} to {args.end_idx-1}", flush=True)

    # Setup
    structures_dir = args.structures_dir
    num_folders = args.end_idx - args.start_idx

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}", flush=True)

    # Ammonium-only dataset:
    # structure_folders = [
    #     os.path.join(os.path.join(structures_dir, top_d), sub_d)
    #     for top_d in os.listdir(structures_dir)
    #     if top_d.startswith("ammonium_") and os.path.isdir(os.path.join(structures_dir, top_d))
    #     for sub_d in os.listdir(os.path.join(structures_dir, top_d))
    #     if os.path.isdir(os.path.join(os.path.join(structures_dir, top_d), sub_d))
    # ]
    
    # Electrolytes unsolvated:
    structure_folders = [
        os.path.join(os.path.join(structures_dir, top_d), sub_d)
        for top_d in os.listdir(structures_dir)
        if os.path.isdir(os.path.join(structures_dir, top_d))
        for sub_d in os.listdir(os.path.join(structures_dir, top_d))
        if os.path.isdir(os.path.join(os.path.join(structures_dir, top_d), sub_d))
    ]

    # Electrolytes redox:
    # structure_folders = [d for d in os.listdir(structures_dir) if os.path.isdir(os.path.join(structures_dir, d))]

    # orca file and density matrix files always have the same names
    orca_file = 'orca.out'
    density_mat_file = 'density_mat.npz'
    dataset_name = 'omol'

    # Basis set configuration
    full_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element]
                  for element in basis_sets.def2_tzvpd.keys()}
    full_basis = dict(sorted(full_basis.items(), key=lambda item: len(item[1]), reverse=True))

    print(f"Job {args.job_id}: Total structure folders available: {len(structure_folders)}", flush=True)

    # Validate indices
    if args.start_idx >= len(structure_folders):
        print(f"Job {args.job_id}: start_idx ({args.start_idx}) >= total folders ({len(structure_folders)}). Nothing to process.", flush=True)
        return

    actual_end_idx = min(args.end_idx, len(structure_folders))
    structures_to_process = structure_folders[args.start_idx:actual_end_idx]

    print(f"Job {args.job_id}: Processing {len(structures_to_process)} structures", flush=True)

    # Initialize storage
    structures = []
    charges = []
    spins = []
    fock_matrix_list = []
    density_matrix_list = []

    orca_output_list = []
    local_folder_name_strings = []
    skipped_structures = []

    big_time_start = time.perf_counter()

    # Process structures
    for folder_idx, structure_folder in enumerate(structures_to_process):
        print(f"Job {args.job_id}: Processing folder {folder_idx}/{len(structures_to_process)}: {structure_folder}", flush=True)

        time_start = time.perf_counter()

        # Read ORCA output
        orca_output_filepath = os.path.join(structures_dir, structure_folder, orca_file)
        print(f"Job {args.job_id}: Reading ORCA output from {orca_output_filepath}", flush=True)
        if not os.path.exists(orca_output_filepath):
            raise FileNotFoundError(f"ORCA output file not found: {orca_output_filepath}", flush=True)

        # General ORCA output file elements:
        print("Parsing ORCA output...")
        parsed_orca_output = utils_orca_out.manually_parse_output(Path(orca_output_filepath), source='manasakani')
        open_shell = parsed_orca_output["unrestricted"]

        # Get elements, coordinates, and matrices for this structure:
        print("Getting fock matrix...")
        fock_matrices, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath, unrestricted=open_shell, get_fock=True)
    
        # Check if the elements exist, or skip this structure:
        if check_elements(elements, full_basis) == False:
            print(f"Skipping {structure_folder}: {elements} not in basis", flush=True)
            continue
        basis = {element: full_basis[element] for element in elements} # Get basis (for this structure) for rearranging the matrix:
        
        print("Getting density matrix...")
        density_output_filepath = os.path.join(structures_dir, structure_folder, density_mat_file)
        density_matrices = np.load(density_output_filepath)
        
        if open_shell:
            alpha_fock_matrix = fock_matrices['alpha']
            beta_fock_matrix = fock_matrices['beta']

            Ptotal = expand_density_mat(density_matrices['orca.scfp'])
            Pspin = expand_density_mat(density_matrices['orca.scfr'])
            alpha_density_matrix = 0.5 * (Ptotal + Pspin)
            beta_density_matrix = 0.5 * (Ptotal - Pspin)
        else:
            fock_matrix = fock_matrices
            density_matrix = expand_density_mat(density_matrices['orca.scfp'])
        
        # process the matri(ces)
        if open_shell:
            alpha_fock_matrix = utils_orca_out.sort_by_m(alpha_fock_matrix, basis, np.array(elements))  # Re-arrange matrix blocks to yzx notation (m=0 is in the middle)
            beta_fock_matrix = utils_orca_out.sort_by_m(beta_fock_matrix, basis, np.array(elements))
            alpha_density_matrix = utils_orca_out.sort_by_m(alpha_density_matrix, basis, np.array(elements))
            beta_density_matrix = utils_orca_out.sort_by_m(beta_density_matrix, basis, np.array(elements))
        else:
            fock_matrix = utils_orca_out.sort_by_m(fock_matrix, basis, np.array(elements))
            density_matrix = utils_orca_out.sort_by_m(density_matrix, basis, np.array(elements))

        # Filtering for errors in fock matrix
        if open_shell:
            if (
                alpha_fock_matrix.max().item() > 10000 or
                beta_fock_matrix.max().item() > 10000 or
                math.isnan(alpha_fock_matrix.max().item()) or
                math.isnan(beta_fock_matrix.max().item())
            ):
                print(alpha_fock_matrix.max().item())
                print(beta_fock_matrix.max().item())
                raise ValueError("This open shell node is too big or contains NaN! Orca calculation might be corrupted")
        else:
            if (
                fock_matrix.max().item() > 10000 or
                math.isnan(fock_matrix.max().item())
            ):
                print(matrix.max().item())
                raise ValueError("This closed shell node is too big or contains NaN! Orca calculation might be corrupted")

        structure = Atoms(elements, positions=coordinates)
        print("Number of atoms:", len(elements), flush=True)

        if folder_idx % 10 == 0:
            print(f"Job {args.job_id}: Structure: {structure}", flush=True)

        if open_shell:
            fock_matrix = [alpha_fock_matrix, beta_fock_matrix]
            density_matrix = [alpha_density_matrix, beta_density_matrix]

        # Store successful structure
        structures.append(structure)
        fock_matrix_list.append(fock_matrix)
        density_matrix_list.append(density_matrix)
        orca_output_list.append(parsed_orca_output)
        local_folder_name_strings.append(structure_folder)

        time_end = time.perf_counter()
        print(f"Job {args.job_id}: Total time for one structure: {time_end - time_start}", flush=True)

        current_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Current: {current_mem:.2f} MB, Peak: {peak_mem:.2f} MB")
        

    big_time_end = time.perf_counter()
    successful_structures = len(structures)
    total_attempted = len(skipped_structures) + successful_structures

    print(f"Job {args.job_id}: Successfully processed {successful_structures} out of {total_attempted} structures", flush=True)
    print(f"Job {args.job_id}: Skipped {len(skipped_structures)} structures due to errors", flush=True)
    if skipped_structures:
        print(f"Job {args.job_id}: Skipped structures: {skipped_structures[:10]}{'...' if len(skipped_structures) > 10 else ''}", flush=True)
    print(f"Job {args.job_id}: Time to process {total_attempted} structures: {big_time_end - big_time_start}", flush=True)

    # Write to database
    output_db_filename = f"{args.output_folder}/{args.output_db_name}_raw_job_{args.job_id}.db"
    print(f"Job {args.job_id}: Writing to {output_db_filename}", flush=True)

    try:
        with connect(output_db_filename) as structure_db:
            print(f"Job {args.job_id}: Writing {len(structures)} structures to database", flush=True)

            for i, (orca_output_dict, structure) in enumerate(zip(orca_output_list, structures)):
                try:
                    if i % 10 == 0:
                        print(f"Job {args.job_id}: Writing structure {i}/{len(structures)}", flush=True)

                    data = {
                        "charge": orca_output_dict["total_charge"],
                        "spin_multiplicity": orca_output_dict["spin_multiplicity"],
                        "fock_matrix": fock_matrix_list[i],
                        "density_matrix": density_matrix_list[i],
                        "total_energy [Eh]": orca_output_dict["total_energy [Eh]"],
                        # "gradient [Eh/bohr]": orca_output_dict["gradient [Eh/bohr]"],
                        "is_open_shell": orca_output_dict["unrestricted"],
                        "num_atoms_in_molecule": len(structure.get_atomic_numbers()),
                        "folder_name": local_folder_name_strings[i]
                    }

                    structure_db.write(structure, data=data)

                except Exception as e:
                    print(f"ERROR: Job {args.job_id} could not write structure {i} due to error: {str(e)}", flush=True)
                    continue

        print(f"Job {args.job_id}: Successfully wrote {len(structures)} structures to {output_db_filename}", flush=True)

        # Verify database
        with connect(output_db_filename) as verify_db:
            actual_count = len(verify_db)
            print(f"Job {args.job_id}: Verification - database contains {actual_count} structures", flush=True)

    except Exception as e:
        print(f"ERROR: Job {args.job_id} could not create/write database {output_db_filename}: {str(e)}", flush=True)

    print(f"Job {args.job_id}: Complete!", flush=True)

if __name__ == "__main__":
    main()
