#!/usr/bin/env python3

import argparse
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
import math

# Your existing imports
from ase import Atoms
from ase.db import connect
from fock_utils import utils_orca_out, utils_tensor_decomp, fock_targets, basis_sets

def parse_args():
    parser = argparse.ArgumentParser(description='Create OMOL dataset (array version)')
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

def get_subdirs(parent_dir, n):
    subdirs = []
    with os.scandir(parent_dir) as it:
        for entry in it:
            if entry.is_dir():
                subdirs.append(entry.name)
                if len(subdirs) >= n:
                    break
    return subdirs

def main():
    args = parse_args()

    print(f"Job {args.job_id}: Processing structures {args.start_idx} to {args.end_idx-1}", flush=True)

    # Setup
    structures_dir = args.structures_dir
    # structure_folders = [f for f in os.listdir(structures_dir)
    #                     if os.path.isdir(os.path.join(structures_dir, f)) ]

    # structure_folders = [f for f in os.listdir(structures_dir)
    #                     if len(os.listdir(os.path.join(structures_dir, f))) > 0 and
    #                     os.path.isdir(os.path.join(structures_dir, f)) ]
    num_folders = args.end_idx - args.start_idx
    structure_folders = get_subdirs(structures_dir, num_folders)

    orca_file = 'orca.out'
    cutoff = 6.0
    dataset_name = 'omol'

    # --> whether to make labels for only half the edges (i,j) where i<j, or all edges (i,j) and (j,i)
    half_edges = False

    # Basis set configuration
    full_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element]
                  for element in basis_sets.def2_tzvpd.keys()}
    full_basis = dict(sorted(full_basis.items(), key=lambda item: len(item[1]), reverse=True))

    # Fock matrix analysis parameters
    orbital_starts = None
    orbital_template = None
    req_output_irreps = None
    out_js_list = None
    ls_list = None

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
    orca_output_list = []
    local_folder_name_strings = []
    skipped_structures = []

    big_time_start = time.perf_counter()

    # Process structures
    for folder_idx, structure_folder in enumerate(structures_to_process):
        try:
            # if folder_idx % 10 == 0:
            print(f"Job {args.job_id}: Processing folder {folder_idx}/{len(structures_to_process)}: {structure_folder}", flush=True)

            # # specific for dimers dataset, which has folder name convention Element1_Element2_dist_charge_spin
            # elements_from_folder = structure_folder.split('_')[:2]
            # if int(utils_orca_out.periodic_table[elements_from_folder[0]]) not in hconfbr or int(utils_orca_out.periodic_table[elements_from_folder[1]]) not in hconfbr:
            #     print(f"Skipping {structure_folder}: {elements_from_folder} not in selected elements", flush=True)
            #     continue

            elements_from_folder = structure_folder.split('_')[:2]
            if not all(utils_orca_out.periodic_table[element] in full_basis for element in elements_from_folder):
                print(f"Skipping {structure_folder}: {elements_from_folder} not in basis", flush=True)
                continue
            time_start = time.perf_counter()

            # Read ORCA output
            orca_output_filepath = os.path.join(structures_dir, structure_folder, orca_file)
            if not os.path.exists(orca_output_filepath):
                raise FileNotFoundError(f"ORCA output file not found: {orca_output_filepath}", flush=True)

            # General ORCA output file elements:
            read_time_start = time.perf_counter()
            parsed_orca_output = utils_orca_out.parse_output(Path(orca_output_filepath), source='manasakani')

            open_shell = parsed_orca_output["unrestricted"]
            spin_multiplicity = parsed_orca_output["spin_multiplicity"]
            charge = parsed_orca_output["total_charge"]

            # Atomic and electronic structure:
            if open_shell:
                print("Loading data from open shell calculation...")
                fock_matrices, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath, unrestricted=True)
                basis = {element: full_basis[element] for element in elements} # Get basis (for this structure) for rearranging the matrix:
                alpha_fock_matrix = utils_orca_out.sort_by_m(fock_matrices['alpha'], basis, np.array(elements))  # Re-arrange matrix blocks to yzx notation (m=0 is in the middle)
                beta_fock_matrix = utils_orca_out.sort_by_m(fock_matrices['beta'], basis, np.array(elements))
            else:
                fock_matrix, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath)
                basis = {element: full_basis[element] for element in elements} # Get basis (for this structure) for rearranging the matrix:
                fock_matrix = utils_orca_out.sort_by_m(fock_matrix, basis, np.array(elements))  # Re-arrange matrix blocks to yzx notation (m=0 is in the middle)

            # Filtering for errors in Fock matrix
            if open_shell:
                if (
                    alpha_fock_matrix.max().item() > 1000 or
                    beta_fock_matrix.max().item() > 1000 or
                    math.isnan(alpha_fock_matrix.max().item()) or
                    math.isnan(beta_fock_matrix.max().item())
                ):
                    print(alpha_fock_matrix.max().item())
                    print(beta_fock_matrix.max().item())
                    raise ValueError("This open shell node is too big or contains NaN! Orca calculation might be corrupted")
            else:
                if (
                    fock_matrix.max().item() > 1000 or
                    math.isnan(fock_matrix.max().item())
                ):
                    print(fock_matrix.max().item())
                    raise ValueError("This closed shell node is too big or contains NaN! Orca calculation might be corrupted")


            #NOTE: The basis returned by utils_orca_out (taken from the output file) is not in the right order for the diffuse functions! So we don't use it directly.
            structure = Atoms(elements, positions=coordinates)
            read_time_end = time.perf_counter()

            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Time to make atoms and get matrix: {read_time_end - read_time_start}", flush=True)
                print(f"Job {args.job_id}: Structure: {structure}", flush=True)

            # Create fock targets:
            target_time_start = time.perf_counter()

            if open_shell:
                fock_matrix = [alpha_fock_matrix, beta_fock_matrix]

            fock_target = fock_targets.Fock_Targets(structure, cutoff, full_basis, dataset_name='omol',
                                                    charge=charge,
                                                    spin_multiplicity=spin_multiplicity,
                                                    fock_matrix=fock_matrix, half_edges=half_edges,
                                                    dtype=torch.float32,
                                                    orbital_starts=orbital_starts,
                                                    orbital_template=orbital_template,
                                                    req_output_irreps=req_output_irreps,
                                                    out_js_list=out_js_list,
                                                    ls_list=ls_list)

            # Save the analysis objects to use for the next structure (these depend only on the basis)
            orbital_starts = fock_target.orbital_starts
            orbital_template = fock_target.orbital_template
            req_output_irreps = fock_target.req_output_irreps
            out_js_list = fock_target.out_js_list
            ls_list = fock_target.ls_list

            target_time_end = time.perf_counter()

            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Time to make targets: {target_time_end - target_time_start}", flush=True)

            # Shift back diffuse orbitals
            for atom, orbitals in full_basis.items():
                full_basis[atom] = [orb % 10 for orb in orbitals]

            # Store successful structure
            structures.append(fock_target)
            orca_output_list.append(parsed_orca_output)
            local_folder_name_strings.append(structure_folder)

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
    output_db_filename = f"omol_dimers/{args.output_db_name}_job_{args.job_id}.db"
    # output_db_filename = f"test.db"
    print(f"Job {args.job_id}: Writing to {output_db_filename}", flush=True)

    try:
        with connect(output_db_filename) as structure_db:
            print(f"Job {args.job_id}: Writing {len(structures)} structures to database", flush=True)

            for i, (orca_output_dict, structure) in enumerate(zip(orca_output_list, structures)):
                try:
                    if i % 10 == 0:
                        print(f"Job {args.job_id}: Writing structure {i}/{len(structures)}", flush=True)

                    atoms = structure.atoms
                    local_folder_name = local_folder_name_strings[i]

                    # It's open shell if it contains two focks
                    is_open_shell = True if structure.node_labels.shape[0] == 2 else False

                    data = {
                        "pos": structure.atoms.get_positions(),
                        "atomic_numbers": structure.atomic_numbers,
                        "charge": structure.charge,
                        "spin_multiplicity": structure.spin_multiplicity,
                        "edge_index": structure.neighbour_list,
                        "edge_mask": structure.forward_edge_mask,
                        "reverse_edge_map": structure.reverse_edge_map,
                        "edge_dist": structure.edge_dist.detach().cpu().numpy(),
                        "node_labels": structure.node_labels.detach().cpu().numpy(),
                        "edge_labels": structure.edge_labels.detach().cpu().numpy(),
                        "total_energy [Eh]": orca_output_dict["total_energy [Eh]"],
                        "gradient [Eh/bohr]": orca_output_dict["gradient [Eh/bohr]"],
                        "half_edges": structure.half_edges,
                        "is_open_shell": is_open_shell,
                        "cutoff": cutoff,
                        "num_atoms_in_molecule": len(structure.atomic_numbers),
                        "folder_name": local_folder_name
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
