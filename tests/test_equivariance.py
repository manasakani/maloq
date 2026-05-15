import os
import torch
import numpy as np
import pickle
from e3nn import o3
from torch_geometric.data import Data, Batch

# --- Project Specific Imports ---
from helm.esen_osh import eSEN_Backbone, Fock_Irreps_Head, HELM_Force_Head, HELM_Energy_Head
from fock_utils import fock_targets_batched, utils_tensor_decomp

def build_batch_nabla(pos_cart, z, edge_index, device, R_cart=None):
    """
    Constructs the gnnData/Batch object.
    Applies the [dist, z, x, y] permutation expected by the eSEN backbone.
    """
    # 1. Calculate displacement in Cartesian space [x, y, z]
    # Edge: src -> dst (edge_index[1] -> edge_index[0] or vice versa per your convention)
    diff = pos_cart[edge_index[1]] - pos_cart[edge_index[0]]
    
    # 2. Permute Cartesian [dx, dy, dz] -> [dz, dx, dy] 
    # xyz (indices 1,2,3) -> yzx -> xyz swap logic
    # To get [dz, dx, dy] from [dx, dy, dz], we use indices [2, 0, 1]
    net_diff = diff[:, [2, 0, 1]]
    dist = torch.norm(diff, dim=-1, keepdim=True)
    
    # edge_attr is [dist, dz, dx, dy]
    edge_attr = torch.cat([dist, net_diff], dim=-1)
    
    data = Data(
        pos=pos_cart,
        z=z.long(),
        atomic_numbers=z.long(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_atoms_in_molecule=torch.tensor([len(z)], device=device),
        charge=torch.zeros(len(z), device=device, dtype=torch.long),
        spin_multiplicity=torch.ones(len(z), device=device, dtype=torch.long)
    )
    return Batch.from_data_list([data]).to(device)

def test_equivariance():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    rcut = 5.0
    lmax = 2
    sphere_channels = 8
    
    # Example orbital basis (def2-svp)
    # This should match basis_sets.orbital_basis_def2_svp_nabla
    orbital_basis = {1: [0], 6: [0, 0, 1], 7: [0, 0, 1], 8: [0, 0, 1]} 

    # --- 1. Mock/Load Data ---
    # Can replace this with actual database[i] call
    # z: [N], pos: [N, 3], ham: [Norb, Norb]
    num_atoms = 3 # Water-like example
    z_sample = torch.tensor([8, 1, 1], device=device)
    pos_sample = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.7, 0.5],
        [0.0, -0.7, 0.5]
    ], device=device)
    
    # For testing purposes, generate a symmetric dummy Hamiltonian 
    # In practice, can load this from the .db file
    norb = 13 # Approx for H2O in svp
    ham_sample = torch.randn(norb, norb, device=device)
    ham_sample = (ham_sample + ham_sample.T) / 2.0 

    # --- 2. Decompose Hamiltonian via Fock_Targets ---
    # This generates the Ground Truth irrep labels (y and node_y)
    print("Decomposing Hamiltonian into Irreps...")
    ft = fock_targets_batched.Fock_Targets(
        [z_sample.cpu().numpy()], 
        [pos_sample.cpu().numpy()], 
        rcut, orbital_basis, 
        [ham_sample.cpu().numpy()],
        dataset_name='nabla_test',
        dtype=dtype
    )
    
    target_irreps = ft.req_output_irreps
    ls_list = ft.ls_list
    edge_index = torch.tensor(ft.neighbour_list_list[0], device=device)
    
    # Ground Truth Labels (Original Orientation)
    y_node_gt = ft.node_labels_list[0][0].to(device) # [num_nodes, irreps_dim]
    y_edge_gt = ft.edge_labels_list[0][0].to(device) # [num_edges, irreps_dim]

    # --- 3. Setup Models ---
    backbone = eSEN_Backbone(
        target_irreps,
        sphere_channels=sphere_channels,
        hidden_channels=sphere_channels,
        lmax=lmax, mmax=lmax,
        include_edges=True
    ).to(device).eval()

    fock_head = Fock_Irreps_Head(
        irreps_in=o3.Irreps([(sphere_channels, (l, 1)) for l in range(lmax + 1)]),
        irreps_out=target_irreps,
        lmax=lmax,
        sphere_channels=sphere_channels,
        ls_list=ls_list,
        reduce_node=True
    ).to(device).eval()

    # --- 4. Rotation Setup ---
    R_cart = o3.rand_matrix().to(dtype).to(device)
    D_wigner = target_irreps.D_from_matrix(R_cart.cpu()).to(device)
    
    # --- 5. Equivariance Check ---
    print(f"Running Equivariance check on {target_irreps}...")

    # Pass A: Original Geometry
    b_orig = build_batch_nabla(pos_sample, z_sample, edge_index, device)
    with torch.no_grad():
        out_orig = backbone(b_orig)
        pred_node_orig, pred_edge_orig = fock_head(out_orig, b_orig)

    print("Original Node Irreps (first node):", pred_node_orig[0].cpu().numpy())

    # Pass B: Rotated Geometry
    pos_rot = pos_sample @ R_cart.T
    b_rot = build_batch_nabla(pos_rot, z_sample, edge_index, device, R_cart=R_cart)
    with torch.no_grad():
        out_rot = backbone(b_rot)
        pred_node_rot, pred_edge_rot = fock_head(out_rot, b_rot)

    print("Rotated Node Irreps (first node):", pred_node_rot[0].cpu().numpy())

    # Verification: Pred(Rot(X)) == D_Wigner * Pred(X)
    expected_node_rot = pred_node_orig @ D_wigner.T
    expected_edge_rot = pred_edge_orig @ D_wigner.T

    print("Expected Rotated Node Irreps (first node):", expected_node_rot[0].cpu().numpy())

    node_err = (expected_node_rot - pred_node_rot).abs().max()
    edge_err = (expected_edge_rot - pred_edge_rot).abs().max()

    print("-" * 30)
    print(f"Fock Node Error: {node_err:.2e}")
    print(f"Fock Edge Error: {edge_err:.2e}")

    # --- 6. Energy/Force Check ---
    # Re-init backbone without edges for Energy/Force (standard HELM setup)
    backbone_ef = eSEN_Backbone(target_irreps, include_edges=False).to(device).eval()
    e_head = HELM_Energy_Head(backbone_ef).to(device).eval()
    f_head = HELM_Force_Head(backbone_ef).to(device).eval()

    with torch.set_grad_enabled(True):
        # Original
        b_ef_orig = build_batch_nabla(pos_sample, z_sample, edge_index, device)
        res_orig = f_head(backbone_ef(b_ef_orig), b_ef_orig)
        f_orig = res_orig['forces']
        e_orig = e_head(backbone_ef(b_ef_orig), b_ef_orig)['energies']

        # Rotated
        b_ef_rot = build_batch_nabla(pos_rot, z_sample, edge_index, device, R_cart=R_cart)
        res_rot = f_head(backbone_ef(b_ef_rot), b_ef_rot)
        f_rot = res_rot['forces']
        e_rot = e_head(backbone_ef(b_ef_rot), b_ef_rot)['energies']

    # Energy is invariant: E_orig == E_rot
    e_err = (e_orig - e_rot).abs().max()
    # Force is equivariant: F_rot == F_orig @ R_cart.T (if force is standard x,y,z)
    # Note: If Force Head outputs in [z, x, y], permute to [x, y, z] first
    f_orig_std = f_orig[:, [1, 2, 0]]
    f_rot_std = f_rot[:, [1, 2, 0]]
    f_err = (f_orig_std @ R_cart.T - f_rot_std).abs().max()

    print(f"Energy Error:    {e_err:.2e}")
    print(f"Force Error:     {f_err:.2e}")
    print("-" * 30)

    if node_err < 1e-7 and edge_err < 1e-7:
        print("✅ SUCCESS: The Model is Equivariant.")
    else:
        print("❌ FAILURE: Equivariance violation detected.")

if __name__ == "__main__":
    test_equivariance()