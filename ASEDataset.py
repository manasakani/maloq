import ase.db
from torch_geometric.data import Data, Dataset, DataLoader
import torch
import numpy as np

class ASEDataset(Dataset):
    def __init__(self, db_path, dtype=torch.float32):
        self.db = ase.db.connect(db_path)
        self.ids = [row.id for row in self.db.select()]
        self.dtype = dtype

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get structure by id
        structure = self.db.get(self.ids[idx])
        
        # Extract atom positions and atomic numbers
        atoms = structure.toatoms()
        positions = atoms.positions
        atomic_numbers = atoms.numbers
        
        # Convert numpy arrays from structure.data to PyTorch tensors
        neighbour_list = structure.data['neighbour_list']
        orbital_basis = structure.data['orbital_basis']
        fock_matrix = torch.tensor(structure.data['fock_matrix'], dtype=self.dtype)
        node_labels = torch.tensor(structure.data['node_labels'], dtype=self.dtype)
        edge_labels = torch.tensor(structure.data['edge_labels'], dtype=self.dtype)
        edge_dist = torch.tensor(structure.data['edge_dist'], dtype=self.dtype)
        edge_index = torch.tensor(neighbour_list, dtype=torch.long)
            
        # Create PyTorch Geometric Data object
        data = Data(
            pos=torch.tensor(positions, dtype=torch.float),
            edge_index=edge_index,
            x=node_labels,
            edge_attr=edge_labels,
            edge_dist=edge_dist,
            fock_matrix=fock_matrix,
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),  
            nedges=len(edge_index[0]), 
            natoms=len(atomic_numbers),  
        )
        
        # Store orbital basis (dictionary)
        data.orbital_basis = orbital_basis
        
        return data