#!/usr/bin/env python3

import argparse
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

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

def main():
    args = parse_args()
    
    print(f"Job {args.job_id}: Processing structures {args.start_idx} to {args.end_idx-1}", flush=True)
    
    # Setup
    structures_dir = args.structures_dir 
    structure_folders = [f for f in os.listdir(structures_dir) 
                        if len(os.listdir(os.path.join(structures_dir, f))) > 0 and 
                        os.path.isdir(os.path.join(structures_dir, f)) ]
    
    orca_file = 'orca.out'
    cutoff = 8.0            
    dataset_name = 'omol'
    
    # --> whether to scale and shift scalar values in the node blocks of the dataset (scale_shift_file needs to be precomputed)
    scale_and_shift = False
    scale_shift_file = 'element_scale_shifts_' + dataset_name + '.pt'
    
    # --> whether to make labels for only half the edges (i,j) where i<j, or all edges (i,j) and (j,i)
    half_edges = False
    
    # Basis set configuration
    full_basis = {utils_orca_out.periodic_table[element]: basis_sets.def2_tzvpd[element] 
                  for element in basis_sets.def2_tzvpd.keys()}
    full_basis = dict(sorted(full_basis.items(), key=lambda item: len(item[1]), reverse=True))
    
    # Fock matrix analysis parameters
    equivariant_blocks = None
    orbital_starts = None
    basis_transformation = None
    
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
            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Processing folder {folder_idx}/{len(structures_to_process)}: {structure_folder}", flush=True)
            
            time_start = time.perf_counter()
            
            # Read ORCA output
            orca_output_filepath = os.path.join(structures_dir, structure_folder, orca_file)
            if not os.path.exists(orca_output_filepath):
                raise FileNotFoundError(f"ORCA output file not found: {orca_output_filepath}", flush=True)
            
            # General ORCA output file elements:
            parsed_orca_output = utils_orca_out.parse_output(Path(orca_output_filepath), source='manasakani')
            
            # Atomic and electronic structure:
            read_time_start = time.perf_counter()
            fock_matrix, elements, coordinates, _ = utils_orca_out.read_orca_out(orca_output_filepath) 
            #NOTE: The basis returned by utils_orca_out (taken from the output file) is not in the right order for the diffuse functions! So we don't use it directly.
            
            # Get basis (for this structure) for rearranging the matrix:
            basis = {element: full_basis[element] for element in elements} 
            fock_matrix = utils_orca_out.sort_by_m(fock_matrix, basis, np.array(elements))  # Re-arrange matrix blocks to yzx notation (m=0 is in the middle)
            
            structure = Atoms(elements, positions=coordinates)  
            read_time_end = time.perf_counter()
            
            if folder_idx % 10 == 0:
                print(f"Job {args.job_id}: Time to make atoms and get matrix: {read_time_end - read_time_start}", flush=True)
                print(f"Job {args.job_id}: Structure: {structure}", flush=True)
            
            # Create fock targets:
            target_time_start = time.perf_counter()
            
            if scale_and_shift:
                print("Getting scale and shift factors...", flush=True)
                print(f"Scale and shift file: {scale_shift_file}", flush=True)
                if scale_shift_file not in os.listdir('./fock_datasets/'):
                    print("[Computing element scale and shift factors for the dataset]", flush=True)
                    get_scale_shift.get_scale_shift(database, dataset_name, rcut_orbitals, dtype=dtype, reduce_edge=reduce_edge)
                else:
                    print("[Loading element scale and shift factors from file]", flush=True)
                    scale_shift_data = torch.load('./fock_datasets/' + scale_shift_file)
                    scale_shift_data = {
                        "element_scalar_means": scale_shift_data["element_scalar_means"],  # dict[int -> list[float]]
                        "element_scalar_stds": scale_shift_data["element_scalar_stds"],    # dict[int -> list[float]]
                        "scalar_irrep_indices": scale_shift_data["scalar_irrep_indices"]   # list[int]
                    }
            else:
                scale_shift_data = None
            
            fock_target = fock_targets.Fock_Targets(structure, cutoff, full_basis, fock_matrix, half_edges=half_edges, scale_shift_data=scale_shift_data, 
                                                        dtype=torch.float32, 
                                                        equivariant_blocks=equivariant_blocks,
                                                        orbital_starts=orbital_starts,
                                                        basis_transformation=basis_transformation)
            
            # Save the analysis objects to use for the next structure (these depend only on the basis)
            equivariant_blocks = fock_target.equivariant_blocks
            orbital_starts = fock_target.orbital_starts
            basis_transformation = fock_target.basis_transformation
            
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
    output_db_filename = f"/checkpoint/ocp/manasakani/{args.output_db_name}_job_{args.job_id}.db"
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
                    
                    data = {
                        "pos": structure.atoms.get_positions(),
                        "atomic_numbers": structure.atomic_numbers,
                        "edge_index": structure.neighbour_list,
                        "edge_mask": structure.forward_edge_mask,
                        "reverse_edge_map": structure.reverse_edge_map,
                        "edge_dist": structure.edge_dist.detach().cpu().numpy(),
                        "node_labels": structure.node_labels.detach().cpu().numpy(),
                        "edge_labels": structure.edge_labels.detach().cpu().numpy(),
                        "total_energy [Eh]": orca_output_dict["total_energy [Eh]"],
                        "gradient [Eh/bohr]": orca_output_dict["gradient [Eh/bohr]"],
                        "half_edges": structure.half_edges,
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
