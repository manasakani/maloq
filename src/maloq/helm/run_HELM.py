# Helper functions to run the HELM model - NEEDS TO BE REFACTORED!!
import torch
import numpy as np
from torch_geometric.data import Data as gnnData
from ..fock_utils import fock_targets, utils_tensor_decomp, matrix2labels_kernels
from e3nn.o3 import Irreps
from .esen_new import eSEN_Backbone, Fock_Irreps_Head
from ase.neighborlist import NeighborList

def make_graph(atoms):
    """
    Make a graph object to input to HELM
    """

    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()

    rcut = 10.0                     # connectivity cutoff (=2xrcut)
    dtype = torch.float64

    num_atoms = len(atoms)
    neighbours = NeighborList(np.ones(num_atoms)*rcut, skin=0, self_interaction=False, bothways=True)
    neighbours.update(atoms)
    neighbour_list = neighbours.get_connectivity_matrix(sparse=True).tocoo()
    neighbour_list = np.vstack([neighbour_list.row, neighbour_list.col])

    # --> Edge distances
    indices0 = neighbour_list[0]  # First atom indices
    indices1 = neighbour_list[1]  # Second atom indices
    edge_dist = torch.zeros((len(indices0), 4), dtype=dtype)
    edge_dist[:, 1:4] = torch.from_numpy(atoms.get_distances(indices1, indices0, vector=True))    # Vector components
    edge_dist[:, 0] = torch.linalg.norm(edge_dist[:, 1:4], dim=-1, keepdim=False)                 # Scalar distances

    atom_graph = gnnData(
                    pos=torch.tensor(positions, dtype=dtype),
                    edge_index=torch.tensor(neighbour_list),
                    edge_attr=edge_dist,
                    atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long).cpu(),
                    num_atoms_in_molecule=num_atoms,
                    nedges=len(neighbour_list[0]),
                )
    return atom_graph

def load_models(backbone_checkpoint, head_checkpoint, orbital_basis, dataset_name='nablaDFT', node_ref_file=None):

    device = torch.device('cuda')
    dtype = torch.float64

    # --> Model settings:
    l_embedding_dim = 128                   # sphere channels
    num_distance_basis = 128                # number of gaussian basis functions used to expand the edge distance
    hidden_dim = l_embedding_dim
    num_mp_layers = 3
    rcut_orbitals = 10.0                     # connectivity cutoff (=2xrcut)
    rcut_gaussian = rcut_orbitals*2         # connectivity cutoff (=2xrcut)
    gaussian_width = 1.0                    # width of gaussians used to expand edge distance
    basis = 'def2-svp'
    functional = 'wb97x-d'
    reduce_node = True                      # inter-orbital forward/backward interactions are enforced to be equal
    reduce_node_intra = True                # intra-orbital interactions are enforced to have 0 odd degrees
    cache_path = "orbital_cache_"+str(dataset_name)+".pkl"

    # --> Orbital analysis for Fock head (read from cache instead!):
    targets, required_irreps, simplified_out_irreps, ls_list, out_js_list, orbital_starts, full_orb_interaction_list = utils_tensor_decomp.make_output_irreps(orbital_basis)
    equivariant_blocks = utils_tensor_decomp.process_targets(orbital_basis, targets, ls_list, out_js_list, full_orb_interaction_list)
    orbital_template = matrix2labels_kernels.get_orbital_template(equivariant_blocks, orbital_starts)
    basis_transformation = utils_tensor_decomp.e3TensorDecomp(required_irreps,
                                                                out_js_list,
                                                                default_dtype_torch=dtype,
                                                                if_sort=False,
                                                                device_torch=device)

    # ls list will define the max basis needed (eg, for OMOL: tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 1, 2])
    ls_list = []
    for l in range(20): # large to account for possible diffuse functions which are incremented by 10
        counts = [torch.sum(torch.tensor(orbital_basis[el]) == l) for el in orbital_basis]
        max_count = max(counts).item()
        ls_list.append(torch.tensor(max_count * [l], dtype=torch.int))

    # Shift back all the diffuse orbitals (which were incremented by 10 in utils_tensor_decomp.py)
    for atom, orbitals in orbital_basis.items():
        orbital_basis[atom] = [orb % 10 for orb in orbitals]

    for atom, orbitals in orbital_basis.items():
        orbital_basis[atom] = [orb % 10 for orb in orbitals]

    ls_list = torch.cat(ls_list)        # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2].
    ls_list = ls_list % 10         # for OMOL: tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 1, 2]

    # --> Initialize backbone and output head:
    irreps_in = Irreps([(l_embedding_dim, (l, 1)) for l in range(required_irreps.lmax + 1)])
    backbone = eSEN_Backbone(
                required_irreps,
                sphere_channels=l_embedding_dim,
                hidden_channels=hidden_dim,
                lmax=required_irreps.lmax,
                mmax=required_irreps.lmax,
                use_pbc=False,
                cutoff=rcut_gaussian,
                edge_channels=l_embedding_dim,
                num_layers=num_mp_layers,
                act_type='gate',
                mlp_type = 'spectral',
                num_distance_basis=num_distance_basis,
                gaussian_width=gaussian_width,
                include_edges=True
            )

    head = Fock_Irreps_Head(irreps_in=irreps_in,
                            irreps_out=required_irreps,
                            lmax=required_irreps.lmax,
                            sphere_channels=l_embedding_dim,
                            half_edges=False,
                            head_type='gated',
                            ls_list=ls_list,
                            reduce_node=reduce_node,
                            reduce_node_intra=reduce_node_intra,
                            orbital_basis=orbital_basis)

    backbone = backbone.to(device)
    head = head.to(device)

    print("Restarting backbone model from :", backbone_checkpoint, flush=True)
    print("Restarting output head model from :", head_checkpoint, flush=True)

    checkpoint = torch.load(backbone_checkpoint)
    state_dict = checkpoint['model_state_dict']

    # Get rid of module prefix if saved with DDP
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    backbone.load_state_dict(new_state_dict)

    checkpoint = torch.load(head_checkpoint)
    state_dict = checkpoint['model_state_dict']

    # Get rid of module prefix if saved with DDP
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    head.load_state_dict(new_state_dict)

    # Scale and shift the orbital self-interaction scalar components of the dataset
    if node_ref_file:
        print("Getting scale and shift factors...", flush=True)
        scale_shift_data = torch.load(node_ref_file)
    else:
        print("Not scaling or shifting the dataset")
        scale_shift_data = None

    HELM = {'backbone': backbone,
            "head": head,
            "basis": basis,
            "functional": functional,
            "orbital_basis": orbital_basis,
            "basis_transform": basis_transformation,
            "orbital_template": orbital_template,
            "node_scale_shifts": scale_shift_data}

    return HELM


def run_HELM_fock(atom_graph, HELM):

    HELM['backbone'].eval()
    HELM['head'].eval()
    dtype = torch.float64

    device = torch.device('cuda')
    atom_graph.to(device)

    with torch.no_grad():
        backbone_out = HELM['backbone'](atom_graph)
        node_output, edge_output  = HELM['head'](backbone_out, atom_graph)

    atom_graph = atom_graph.cpu()
    atomic_numbers = atom_graph.atomic_numbers
    num_atoms = len(atomic_numbers)

    # Scalar irrep referencing
    if HELM['node_scale_shifts']:
        new_node_blocks = node_output.clone()
        means = HELM['node_scale_shifts']['element_scalar_means']
        stds = HELM['node_scale_shifts']['element_scalar_stds']
        scalar_indices = HELM['node_scale_shifts']['scalar_irrep_indices']

        for i, (node_block, z) in enumerate(zip(node_output, atomic_numbers)):
            z = int(z.item()) if isinstance(z, torch.Tensor) else int(z)

            mean_vals = means[z]
            std_vals = stds[z]

            for idx_offset, idx in enumerate(scalar_indices):
                new_node_blocks[i][idx] = node_block[idx] * std_vals[idx_offset] + mean_vals[idx_offset]
        node_output = new_node_blocks

    # Basis transformation
    uncoupled_node_outputs = HELM['basis_transform'].get_H(node_output)
    uncoupled_edge_outputs = HELM['basis_transform'].get_H(edge_output)

    orbitals_per_atom = ([ sum([(2*l+1)
                        for l in HELM['orbital_basis'][atom_number]])
                        for atom_number in atomic_numbers.numpy() ])
    block_starts = np.hstack([0, np.cumsum(orbitals_per_atom)])

    matrix_size = block_starts[-1]
    src_idx, target_idx = atom_graph.edge_index[0], atom_graph.edge_index[1]
    src_idxes = np.concatenate([src_idx, np.arange(num_atoms)])
    target_idxes = np.concatenate([target_idx, np.arange(num_atoms)])
    fock_block_offsets = np.concatenate([np.array([0]), np.cumsum(orbitals_per_atom)])

    output_fock_matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    output_targets = np.concatenate([uncoupled_edge_outputs.detach().cpu().numpy(), uncoupled_node_outputs.detach().cpu().numpy()])
    matrix2labels_kernels.numpy_single_matrix2label(
        HELM['orbital_template'],
        fock_block_offsets,
        atomic_numbers,
        src_idxes,
        target_idxes,
        output_fock_matrix,
        output_targets,
        forward=False
    )

    output_fock_matrix = (output_fock_matrix + output_fock_matrix.T) / 2

    return output_fock_matrix
