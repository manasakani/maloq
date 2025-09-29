import os
import glob
import ase.db
from tqdm import tqdm

def count_atoms_edges_in_db(db_path):
    """Count atoms and edges in a single database file and write atoms per structure to a text file"""
    print(f"Processing: {os.path.basename(db_path)}")
    
    db = ase.db.connect(db_path)
    total_atoms = 0
    total_edges = 0
    total_molecules = 0
    atoms_per_structure = []
    
    # Process one row at a time to avoid memory issues
    for row in tqdm(db.select(), desc=f"Processing {os.path.basename(db_path)}"):
        print(f"Row ID: {row.id}", flush=True)
        try:
            # Get atoms object and count atoms directly
            atoms = row.toatoms()
            natoms = len(atoms)
            print(f"natoms from atoms object: {natoms}", flush=True)
            atoms_per_structure.append(natoms)
            # Get edge_index from data and count edges
            if 'edge_index' in row.data:
                edge_index = row.data['edge_index']
                # edge_index is typically a 2xN array where N is number of edges
                if hasattr(edge_index, 'shape') and len(edge_index.shape) == 2:
                    nedges = edge_index.shape[1]  # Number of columns = number of edges
                elif hasattr(edge_index, '__len__') and len(edge_index) >= 2:
                    nedges = len(edge_index[0])  # Length of first row = number of edges
                else:
                    nedges = 0
            else:
                nedges = 0
            
            total_atoms += natoms
            total_edges += nedges
            total_molecules += 1
            
            # Debug: Print first few values
            if total_molecules <= 3:
                print(f"  Row {row.id}: natoms={natoms}, nedges={nedges}")
            
        except Exception as e:
            print(f"Error processing row {row.id}: {e}")
            continue
    
    # Write atoms per structure to a text file
    out_txt = os.path.splitext(os.path.basename(db_path))[0] + "_atoms_per_structure.txt"
    with open(out_txt, "w") as f:
        for n in atoms_per_structure:
            f.write(f"{n}\n")
    print(f"Wrote atoms per structure to {out_txt}")
    
    print(f"  Molecules: {total_molecules:,}")
    print(f"  Atoms: {total_atoms:,}")
    print(f"  Edges: {total_edges:,}")
    print()
    
    return total_molecules, total_atoms, total_edges

def main():
    # Define the database directory pattern
    db_directory = "/checkpoint/ocp/manasakani/omol_test_all_5k/"
    db_pattern = os.path.join(db_directory, "omol_closedshell_58k_test_all_5k_6.0_alledge_job_*.db")
    
    # Find all database files
    db_files = sorted(glob.glob(db_pattern))
    
    if not db_files:
        print(f"No database files found matching pattern: {db_pattern}")
        return
    
    print(f"Found {len(db_files)} database files")
    print("=" * 60)
    
    # Initialize counters
    grand_total_molecules = 0
    grand_total_atoms = 0
    grand_total_edges = 0
    
    # Process each database file
    for db_file in db_files:
        if os.path.exists(db_file):
            molecules, atoms, edges = count_atoms_edges_in_db(db_file)
            grand_total_molecules += molecules
            grand_total_atoms += atoms
            grand_total_edges += edges
        else:
            print(f"File not found: {db_file}")
    
    # Print final results
    print("=" * 60)
    print("FINAL TOTALS:")
    print(f"Total database files processed: {len(db_files)}")
    print(f"Total molecules: {grand_total_molecules:,}")
    print(f"Total atoms: {grand_total_atoms:,}")
    print(f"Total edges: {grand_total_edges:,}")
    print(f"Average atoms per molecule: {grand_total_atoms/grand_total_molecules:.2f}")
    print(f"Average edges per molecule: {grand_total_edges/grand_total_molecules:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()


# Test -1k:
# Total database files processed: 4
# Total molecules: 1,006
# Total atoms: 98,399
# Total edges: 9,140,166
# Average atoms per molecule: 97.81
# Average edges per molecule: 9085.65

# Test -5k:
# Total database files processed: 16
# Total molecules: 4,937
# Total atoms: 370,924
# Total edges: 36,514,364
# Average atoms per molecule: 75.13
# Average edges per molecule: 7396.06

# Train 58k:
# Total database files processed: 64
# Total molecules: 56,657
# Total atoms: 3,362,755
# Total edges: 241,227,574
# Average atoms per molecule: 59.35
# Average edges per molecule: 4257.68