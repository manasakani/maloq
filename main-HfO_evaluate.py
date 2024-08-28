import lib.data as data
import lib.models as models
import lib.training as training
import lib.structure as structure
import lib.utils as utils
import lib.SO2 as SO2
import lib.so2_model as so2_model
import lib.SO3 as SO3
from e3nn.o3 import Irreps
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random


def main():

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    save_file = 'model_test_info_GPU'

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # *** Initialize the domain and electronic structure matrices:

    # *** Preform orbital analysis:
    atom_orbitals = {'8':[0,1],'72':[0,0,1,2]}                                            # Orbital types of each atom in the structure
    no_parity = True                                                                            # No parity symmetry          
    orbital_types = [[0,1],[0,0,1,2]]                                                       # orbital types of each atom in the structure 

    targets, net_out_irreps, net_out_irreps_simplified = SO2.orbital_analysis(atom_orbitals, targets=None, no_parity=no_parity)
    index_to_Z,inverse_indices = torch.unique(torch.tensor([8,72]), sorted=True, return_inverse=True)
    equivariant_blocks, out_js_list, out_slices = SO2.process_targets(orbital_types, index_to_Z, targets)

    # equivariant_blocks: start and end indices of the equivariant blocks in i and j direction for each target in targets
    # out_js_list: ll the l1 l2 interactions needed 
    # out_slices: marks the start and end of indices belonging to a certain target. Slice 1 (0 to 1) corresponds to the first target in equivariant blocks 

    construct_kernel = SO2.e3TensorDecomp(net_out_irreps, 
                                          out_js_list, 
                                          default_dtype_torch= torch.float32, 
                                          spinful=False,
                                          no_parity=no_parity, 
                                          if_sort=False, 
                                          device_torch='cpu') #the data is created on cpu, so the construct_kernel must be on cpu 

    model = torch.load('model_test_GPU.pt')
    test_batch = torch.load('model_test_GPU_batch_structure_0_training_1000_1.5.pt')

    # *** Evaluate the model:
    MAE_node, MAE_edge =  training.evaluate_model(model, test_batch, construct_kernel, equivariant_blocks, atom_orbitals, out_slices, device, save_file=save_file)

if __name__ == "__main__":
    main()