import time
import_start = time.perf_counter()
import os, sys
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
import utils_orca_out, fock_targets
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist

from ASEDataset import ASEDataset, ASEAtomsData, sampleDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as gnnData, Dataset
import random

from equiformer.network import SO2Net
from equiformer.SO3 import CoefficientMappingModule

from esen.esen import eSEN_Backbone
from e3nn.o3 import Irreps, rand_matrix
import utils_training
import_end = time.perf_counter()
print("Time to do imports: ", import_end - import_start)

def custom_collate_fn(batch):
    return Batch.from_data_list(batch)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Fix this later:
sys.path.append('/home/manasakani/fairchem/src/')

# Settings (just dumping everything here for now)
# -----------------------------------------------
dbpath = 'fock_datasets/schnorb_hamiltonian_water.db'
database = ASEAtomsData(dbpath)
print("Targets available: ", database.available_properties)

# -> Model settings:
l_embedding_dim = 128                   # sphere channels
num_distance_basis = 128                # number of gaussian basis functions used to expand the edge distance
hidden_dim = 128
cutoff = 6.0*2                          # Cutoff used for edge distance embedding
num_mp_layers = 1
model_name = 'esen'
restart = False
output_folder = 'outputs_QM7'
loss_fxn = utils_training.mse_padded_loss

# -> Training settings:
num_val = 1                           # Number of validation structures
num_train = 1
dtype = torch.float32
batch_size = 100

# --> Compute env
device = torch.device('cuda')         
world_size = int(os.environ['SLURM_NTASKS'])
rank = int(os.environ['SLURM_PROCID'])
dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
torch.cuda.set_device(0) # visibility is restricted to 0 in .sh file

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Prepare data
# --------------------------------------------
data_load_start = time.perf_counter()
max_mol = 5000 
num_molecules = num_val + num_train
random_indices = random.sample(range(num_molecules), min(max_mol, num_molecules))

datalist = []
for i in range(num_molecules):      # deterministic
    mol = database.__getitem__(0)   # get the same molecule twice for this test, the one that the network was overtrained on

    mol_atoms = Atoms(symbols=mol['_atomic_numbers'].numpy(), positions=mol['_positions'].numpy())
    rcut = 100.0                                            # connectivity cutoff
    num_atoms = len(mol['_positions'])
    energy = mol['energy']
    forces = mol['forces']

    print("molecule positions: ", mol['_positions'].numpy())

    # Electronic structure matrix:
    hamiltonian = mol['hamiltonian'].numpy()   
    orbital_basis = {8: [0, 0, 0, 1, 1, 2], 1: [0, 0, 1]}
    atomic_numbers = mol['_atomic_numbers'].numpy()
    hamiltonian = utils_orca_out.sort_by_m(hamiltonian, orbital_basis, atomic_numbers)  

    time_start = time.perf_counter()
    graph_targets = fock_targets.Fock_Targets(mol_atoms, rcut, orbital_basis, hamiltonian)
    time_end = time.perf_counter()
    print("time to make targets: ", time_end - time_start)

    data = gnnData(
                    pos=torch.tensor(graph_targets.atoms.positions, dtype=torch.float),
                    x=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long), 
                    edge_index=torch.tensor(graph_targets.neighbour_list).to(device),
                    edge_attr=graph_targets.edge_dist.to(device),
                    y=graph_targets.edge_labels,
                    node_y=graph_targets.node_labels,
                    atomic_numbers=torch.tensor(graph_targets.atomic_numbers, dtype=torch.long).cpu(),  
                    energies=torch.tensor(energy, dtype=dtype),
                    forces=torch.tensor(forces, dtype=dtype),
                )
    datalist.append(data)

required_irreps = graph_targets.req_output_irreps
print("required irreps: ", required_irreps)

train_size = len(datalist) - num_val
train_datalist, val_datalist = torch.utils.data.random_split(datalist, [train_size, num_val])
train_dataset = sampleDataset(train_datalist)
val_dataset = sampleDataset(val_datalist)

# Check use of batch size higher than 1!!
train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=False, num_workers=0)
data_load_end = time.perf_counter()
print("Time to load dataset: ", data_load_end - data_load_start)

# Prepare model
# --------------------------------------------
model_setup_start = time.perf_counter()
if model_name == 'equiformer':
    irreps_in = Irreps([(l_embedding_dim, (l, 1)) for l in range(required_irreps.lmax + 1)]) 
    mappingReduced = CoefficientMappingModule(required_irreps.lmax, required_irreps.lmax)
    edge_channels_list = [l_embedding_dim, l_embedding_dim, l_embedding_dim]

    attn_hidden_channels = 128 
    attn_alpha_channels = 32
    attn_value_channels = 32 
    ffn_hidden_channels = 64 
    num_heads=2

    model = SO2Net(num_mp_layers, 
                    required_irreps.lmax, 
                    required_irreps.lmax, 
                    mappingReduced, 
                    l_embedding_dim, 
                    edge_channels_list, 
                    attn_hidden_channels, 
                    num_heads, 
                    attn_alpha_channels, 
                    attn_value_channels, 
                    ffn_hidden_channels, 
                    irreps_in, 
                    required_irreps)
elif model_name == 'esen':
    model = eSEN_Backbone(
                    required_irreps,
                    sphere_channels=l_embedding_dim,
                    hidden_channels=hidden_dim,
                    lmax=required_irreps.lmax,
                    mmax=required_irreps.lmax,
                    use_pbc=False,
                    cutoff=cutoff,
                    edge_channels=l_embedding_dim,
                    num_layers=num_mp_layers,
                    act_type='gate',
                    mlp_type = 'spectral',
                    num_distance_basis=num_distance_basis
                )

model = model.to(device)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
model_setup_end = time.perf_counter()
print("Time to setup model: ", model_setup_end - model_setup_start)

if restart:
    print("Restarting model from ", output_folder)
    restart_file = output_folder + "/model.pt.pt"
    checkpoint = torch.load(restart_file)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)

# Rotation matrices:
cartesian_rot_mat = rand_matrix(dtype=dtype)
spherical_rot_mat = required_irreps.D_from_matrix(cartesian_rot_mat).to(device)
cartesian_rot_mat = cartesian_rot_mat.to(device)

# Check equivariance of the network itself:
# -----------------------------------------
for batch in train_loader:
    model.eval()

    batch = batch.to(device)
    rotated_input = batch
    rotated_output = deepcopy(batch)

    # --> 1. Rotate the input of the "rotated input" batch - f(R(x)):
    rotated_input.pos = (cartesian_rot_mat @ rotated_input.pos.T).T                 # this doesnt actually do anything though
    
    print("initial rotated_input.edge_attr:", rotated_input.edge_attr)
    rotated_input_edge_dist = rotated_input.edge_attr                               # this is what the network will see
    rotated_input_edge_vec = rotated_input_edge_dist[:, [2, 3, 1]]                  # xyz -> yzx
    rotated_input_edge_vec = (cartesian_rot_mat @ rotated_input_edge_vec.T).T       
    rotated_input_edge_vec = rotated_input_edge_vec[:, [2, 0, 1]]                   # yzx -> xyz because the network will re-permute           
    rotated_input.edge_attr[:, 1:4] = rotated_input_edge_vec
    print("final rotated_input.edge_attr: ", rotated_input.edge_attr)

    # with torch.no_grad():
    rotated_input = model(rotated_input) 
    rotated_input_nodes = rotated_input["node_rankN"]
    rotated_input_edges = rotated_input["edge_rankN"]
    
    # --> 2. Rotated the output of the "rotated output" batch - WD(f(x)):
    # with torch.no_grad():
    rotated_output = model(rotated_output) 
    rotated_output_nodes = (spherical_rot_mat @ rotated_output["node_rankN"].T).T
    rotated_output_edges = (spherical_rot_mat @ rotated_output["edge_rankN"].T).T

    print("rotated input node 0: ", rotated_input_nodes[0][0:6])
    print("rotated output node 0: ", rotated_output_nodes[0][0:6])

    if not os.path.exists("equivariance_test_results"):
        os.makedirs("equivariance_test_results")

    for n in range(len(rotated_output_nodes)):
        print("Checking node " + str(n) + "...")
        if not torch.allclose(rotated_input_nodes[n], rotated_output_nodes[n]):
            print(f"Error: Node {n} input and output are not allclose")
        plt.imshow(rotated_input_nodes[n].detach().cpu().numpy().reshape(14, 14) 
                - rotated_output_nodes[n].detach().cpu().numpy().reshape(14, 14), vmin=0, vmax=1e-6)
        plt.colorbar()
        plt.savefig("equivariance_test_results/numerical_equivariance_err_node" + str(n) + ".png", dpi=300, bbox_inches='tight')
        plt.close()

    for e in range(len(rotated_output_edges)):
        print("Checking edge " + str(e) + "...")
        if not torch.allclose(rotated_input_edges[e], rotated_output_edges[e]):
            print(f"Error: Edge {e} input and output are not allclose")
        plt.imshow(rotated_input_edges[e].detach().cpu().numpy().reshape(14, 14) 
                - rotated_output_edges[e].detach().cpu().numpy().reshape(14, 14), vmin=0, vmax=1e-6)
        plt.colorbar()
        plt.savefig("equivariance_test_results/numerical_equivariance_err_edge" + str(e) + ".png", dpi=300, bbox_inches='tight')
        plt.close()

    print("Finished numerical equivariance check, now doing model equivariance check")

# Check equivariance of the data pipeline
# -----------------------------------------
for batch in val_loader:
    model.eval()

    batch = batch.to(device)
    unrotated_input = batch
    rotated_input = deepcopy(unrotated_input)


    # --> 1. Rotate the input of the "rotated input" batch - f(R(x)):
    rotated_input.pos = (cartesian_rot_mat @ rotated_input.pos.T).T                 # this doesnt actually do anything though
    rotated_input_edge_dist = rotated_input.edge_attr                               # this is what the network will see
    rotated_input_edge_vec = rotated_input_edge_dist[:, [2, 3, 1]]                  # xyz -> yzx
    rotated_input_edge_vec = (cartesian_rot_mat @ rotated_input_edge_vec.T).T       
    rotated_input_edge_vec = rotated_input_edge_vec[:, [2, 0, 1]]                   # yzx -> xyz because the network will do -> xyz   
    rotated_input.edge_attr[:, 1:4] = rotated_input_edge_vec

    # with torch.no_grad():
    unrotated_mol = model(unrotated_input) 
    rotated_mol = model(rotated_input) 

    unrotated_node_output = unrotated_mol["node_rankN"]
    unrotated_edge_output = unrotated_mol["edge_rankN"]
    rotated_node_output = rotated_mol["node_rankN"]
    rotated_edge_output = rotated_mol["edge_rankN"]
    
    # --> 2. Rotate the corresponding Fock matrix blocks:
    unrotated_node_labels = batch.node_y
    unrotated_edge_labels = batch.y
    rotated_node_labels = (spherical_rot_mat @ unrotated_node_labels.T).T
    rotated_edge_labels = (spherical_rot_mat @ unrotated_edge_labels.T).T

    # --> 3. Check the loss between the rotated and unrotated cases
    print("unrotated model node 0 output: ", unrotated_node_output[0][0:6])
    print("unrotated model node 0 label: ", unrotated_node_labels[0][0:6])
    print("rotated model node 0 output: ", rotated_node_output[0][0:6])
    print("rotated label node 0 label: ", rotated_node_labels[0][0:6])

    unrotated_output = torch.cat([unrotated_node_output, unrotated_edge_output], dim=0)
    unrotated_labels = torch.cat([unrotated_node_labels, unrotated_edge_labels], dim=0)
    unrotated_loss = loss_fxn(unrotated_output, unrotated_labels)

    rotated_output = torch.cat([rotated_node_output, rotated_edge_output], dim=0)
    rotated_labels = torch.cat([rotated_node_labels, rotated_edge_labels], dim=0)
    rotated_loss = loss_fxn(rotated_output, rotated_labels)

    print("Loss between unrotated molecule and unrotated H blocks: ", unrotated_loss)
    print("Loss between rotated molecule and rotated H blocks: ", rotated_loss)

    plt.imshow(np.log(np.abs(unrotated_node_output[0].detach().cpu().numpy().reshape(14, 14))), vmin=-5, vmax=5)
    plt.colorbar()
    plt.savefig("equivariance_test_results/unrotated_node_output[0].png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(np.log(np.abs(unrotated_node_labels[0].detach().cpu().numpy().reshape(14, 14))), vmin=-5, vmax=5)
    plt.colorbar()
    plt.savefig("equivariance_test_results/unrotated_node_labels[0].png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(np.log(np.abs(rotated_node_output[0].detach().cpu().numpy().reshape(14, 14))), vmin=-5, vmax=5)
    plt.colorbar()
    plt.savefig("equivariance_test_results/rotated_node_output[0].png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(np.log(np.abs(rotated_node_labels[0].detach().cpu().numpy().reshape(14, 14))), vmin=-5, vmax=5)
    plt.colorbar()
    plt.savefig("equivariance_test_results/rotated_node_labels[0].png", dpi=300, bbox_inches='tight')
    plt.close()

    # plt.imshow(np.abs( (rotated_node_labels[0].detach().cpu().numpy().reshape(14, 14) 
    #                          - rotated_node_output[0].detach().cpu().numpy().reshape(14, 14))) / rotated_node_output[0].detach().cpu().numpy().reshape(14, 14), vmin=0, vmax=1)
    plt.imshow(np.abs( (rotated_node_labels[0].detach().cpu().numpy().reshape(14, 14) 
                             - rotated_node_output[0].detach().cpu().numpy().reshape(14, 14))), vmin=0, vmax=0.01)
    
    plt.colorbar()
    plt.savefig("equivariance_test_results/rotated_percent_err[0].png", dpi=300, bbox_inches='tight')
    plt.close()
             