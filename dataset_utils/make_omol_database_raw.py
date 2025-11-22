#!/usr/bin/env python3

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# Your existing imports
from ase import Atoms
from ase.db import connect
from fock_utils import utils_orca_out

def parse_args():
    parser = argparse.ArgumentParser(description='Create raw Fock matrix dataset (array version)')
    parser.add_argument('-f', '--structures_dir', type=str, required=True,
                       help='Directory containing structure folders')
    parser.add_argument('-o', '--output_db_name', type=str, required=True,
                       help='Output database name (without .db extension)')
    parser.add_argument('--start_idx', type=int, required=True,
                       help='Start index for structure processing')
    parser.add_argument('--end_idx', type=int, required=True,
                       help='End index for structure processing')
    parser.add_argument('--job_id', type=int, required=True,
                       help='SLURM array job ID')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Job {args.job_id}: Processing structures {args.start_idx} to {args.end_idx-1}", flush=True)

    # Setup
    structures_dir = args.structures_dir
    structure_folders = [f for f in os.listdir(structures_dir)
                        if len(os.listdir(os.path.join(structures_dir, f))) > 0 and
                        os.path.isdir(os.path.join(structures_dir, f)) ]

    orca_file = 'orca.out'

    print(f"Job {args.job_id}: Total structure folders available: {len(structure_folders)}", flush=True)

    # Validate indices
    if args.start_idx >= len(structure_folders):
        print(f"Job {args.job_id}: start_idx ({args.start_idx}) >= total folders ({len(structure_folders)}). Nothing to process.", flush=True)
        return

    actual_end_idx = min(args.end_idx, len(structure_folders))
    structures_to_process = structure_folders[args.start_idx:actual_end_idx]

    print(f"Job {args.job_id}: Processing {len(structures_to_process)} structures", flush=True)

    # Initialize storage
    local_folder_name_strings = []
    structures = []
    fock_matrices = []
    orca_output_list = []
    skipped_structures = []

    big_time_start = time.perf_counter()

    # Process structures
    for folder_idx, structure_folder in enumerate(structures_to_process):
        try:
            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Processing folder {folder_idx}/{len(structures_to_process)}: {structure_folder}", flush=True)

            time_start = time.perf_counter()

            # Read ORCA output
            orca_output_filepath = os.path.join(structures_dir, structure_folder, orca_file)
            if not os.path.exists(orca_output_filepath):
                raise FileNotFoundError(f"ORCA output file not found: {orca_output_filepath}")

            local_folder_name_strings.append(structure_folder)

            # Get data:
            parse_time_start = time.perf_counter()
            parsed_orca_output = utils_orca_out.parse_output(Path(orca_output_filepath), source='manasakani')
            parse_time_end = time.perf_counter()
            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Time to parse orca output: {parse_time_end - parse_time_start}", flush=True)

            fock_time_start = time.perf_counter()
            fock_matrix, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath)
            fock_time_end = time.perf_counter()
            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Time to make fock matrix: {fock_time_end - fock_time_start}", flush=True)

            # Make structure:
            structure_time_start = time.perf_counter()
            structure = Atoms(elements, positions=coordinates)
            structure_time_end = time.perf_counter()
            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Time to make structure: {structure_time_end - structure_time_start}", flush=True)
                print(f"Job {args.job_id}: Structure: {structure}", flush=True)

            flatten_time_start = time.perf_counter()
            assert fock_matrix.ndim > 0
            assert fock_matrix.size > 0
            flattened_fock = fock_matrix[np.triu_indices_from(fock_matrix)]
            flatten_time_end = time.perf_counter()
            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Time to flatten fock: {flatten_time_end - flatten_time_start}", flush=True)

            orca_output_list.append(parsed_orca_output)
            structures.append(structure)
            fock_matrices.append(flattened_fock)

            time_end = time.perf_counter()
            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Total time for one structure: {time_end - time_start}", flush=True)

        except Exception as e:
            print(f"ERROR: Job {args.job_id} skipping structure {structure_folder} due to error: {str(e)}", flush=True)
            skipped_structures.append(structure_folder)
            continue

    big_time_end = time.perf_counter()
    successful_structures = len(structures)
    total_attempted = len(skipped_structures) + successful_structures

    print(f"Job {args.job_id}: Successfully processed {successful_structures} out of {total_attempted} structures", flush=True)
    print(f"Job {args.job_id}: Skipped {len(skipped_structures)} structures due to errors", flush=True)
    if skipped_structures:
        print(f"Job {args.job_id}: Skipped structures: {skipped_structures[:10]}{'...' if len(skipped_structures) > 10 else ''}", flush=True)
    print(f"Job {args.job_id}: Time to process {total_attempted} structures: {big_time_end - big_time_start}", flush=True)

    # Write to database
    # output_db_filename = f"{args.output_db_name}_job_{args.job_id}.db"
    output_db_filename = f"/checkpoint/ocp/manasakani/OMol_CSH_58k/omol_csh_58k_train/{args.output_db_name}_job_{args.job_id}.db"
    print(f"Job {args.job_id}: Writing to {output_db_filename}", flush=True)

    try:
        with connect(output_db_filename) as structure_db:
            print(f"Job {args.job_id}: Writing {len(structures)} structures to database", flush=True)

            for i, (local_folder_name, orca_output_dict, atoms, fock) in enumerate(zip(local_folder_name_strings, orca_output_list, structures, fock_matrices)):
                try:
                    if i % 10 == 0:
                        print(f"Job {args.job_id}: Writing structure {i}/{len(structures)}", flush=True)

                    data = {
                        "total_energy [Eh]": orca_output_dict["total_energy [Eh]"],
                        "gradient [Eh/bohr]": orca_output_dict["gradient [Eh/bohr]"],
                        "Hamiltonian [Eh]": fock,
                        "omol_folder_name": local_folder_name
                    }

                    structure_db.write(atoms, data=data)

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
