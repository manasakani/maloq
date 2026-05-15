import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
    diff = pos_cart[edge_index[1]] - pos_cart[edge_index[0]]
    
    # 2. Permute Cartesian [dx, dy, dz] -> [dz, dx, dy] 
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
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    rcut = 5.0
    sphere_channels = 8
    torch.manual_seed(42)
    
    # Example orbital basis (def2-svp)
    orbital_basis = {1: [0], 8: [0, 0, 1, 2, 3, 4]} 

    # --- 1. Mock/Load Data ---
    z_sample = torch.tensor([8, 1, 1], device=device)
    pos_sample = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 0.7, 0.5],
        [0.0, -0.4, 0.5]
    ], device=device)
    
    norb = 0
    for Z in z_sample:
        orbs = orbital_basis[Z.item()]
        norb += sum([(2*l + 1) for l in orbs])  # s:1, p:3, d:5, f:7, g:9, h:11

    print("Making fake Hamitonian of size:", norb, "x", norb)
    ham_sample = torch.randn(norb, norb, device=device)
    ham_sample = (ham_sample + ham_sample.T) / 2.0 

    # --- 2. Decompose Hamiltonian via Fock_Targets ---
    print("Decomposing Hamiltonian into Irreps...")
    ft = fock_targets_batched.Fock_Targets(
        [z_sample.cpu().numpy()], 
        [pos_sample.cpu().numpy()], 
        rcut, orbital_basis, 
        [ham_sample.cpu().numpy()],
        dataset_name='test',
        dtype=dtype
    )
    
    target_irreps = ft.req_output_irreps
    lmax = max([l for _, (l, _) in target_irreps])
    print("target_irreps:", target_irreps, " Lmax:", lmax)
    ls_list = ft.ls_list
    edge_index = torch.tensor(ft.neighbour_list_list[0], device=device)

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

    backbone_ef = eSEN_Backbone(target_irreps, include_edges=False).to(device).eval()
    e_head = HELM_Energy_Head(backbone_ef).to(device).eval()
    f_head = HELM_Force_Head(backbone_ef).to(device).eval()

    # --- New: Polar Loop Logic ---
    num_angles = 100
    angles = np.linspace(0, 2 * np.pi, num_angles)
    h_errs, f_errs, e_errs = [], [], []

    # Reference pass (Angle 0)
    b_orig = build_batch_nabla(pos_sample, z_sample, edge_index, device)
    with torch.set_grad_enabled(True):
        out_f_orig = backbone(b_orig)
        pred_node_orig, _ = fock_head(out_f_orig, b_orig)
        out_ef_orig = backbone_ef(b_orig)
        f_orig = f_head(out_ef_orig, b_orig)['forces'][:, [1, 2, 0]]
        e_orig = e_head(out_ef_orig, b_orig)['energies']

    print(f"Looping over {num_angles} rotation angles...")
    for theta in angles:
        # Rotation Matrix around Z
        R_cart = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ], dtype=dtype, device=device)
        D_wigner = target_irreps.D_from_matrix(R_cart.cpu()).to(device)
        
        pos_rot = pos_sample @ R_cart.T
        b_rot = build_batch_nabla(pos_rot, z_sample, edge_index, device)
        
        with torch.set_grad_enabled(True):
            # Fock
            node_rot, _ = fock_head(backbone(b_rot), b_rot)
            # Energy/Force
            out_ef_rot = backbone_ef(b_rot)
            e_rot = e_head(out_ef_rot, b_rot)['energies']
            f_rot = f_head(out_ef_rot, b_rot)['forces'][:, [1, 2, 0]]

        # Calculate errors
        h_errs.append((pred_node_orig @ D_wigner.T - node_rot).abs().max().item())
        f_errs.append((f_orig @ R_cart.T - f_rot).abs().max().item())
        e_errs.append((e_orig - e_rot).abs().max().item())

    # --- 4. Plotting ---
    output_dir = 'tests/equivariance_test_results'
    os.makedirs(output_dir, exist_ok=True)

    tolerance_val = -4  # log10(1e-4)
    
    plots = [
        (h_errs, "Hamiltonian Node Equivariance Error", "hamiltonian.png", "royalblue"),
        (f_errs, "Force Equivariance Error", "forces.png", "tomato"),
        (e_errs, "Energy Invariance Error", "energy.png", "forestgreen")
    ]

    for data, title, fname, color in plots:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 4))
        ax.plot(angles, np.log10(np.array(data) + 1e-16), color=color, linewidth=1.5)

        # Add a black dashed tolerance line at 1e-4
        ax.set_ylim(-17, 1)
        ax.plot(np.linspace(0, 2*np.pi, 100), [tolerance_val]*100, 
                color='black', linestyle='--', linewidth=1.0, label='Tolerance (1e-4)')

        ax.set_rticks([-15, -12, -9, -6, -3])
        ax.set_yticklabels(['1e-15', '1e-12', '1e-9', '1e-6', '1e-3'])
        ax.tick_params(axis='y', labelsize=8)  # Radial ticks
        ax.tick_params(axis='x', labelsize=10)  # Angular ticks (0, 45, 90...)

        ax.set_title(title + " (log10)", y=1.1, fontsize=12)

        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    print("Equivariance error (mean):")
    print(f"  Hamiltonian Node: {np.mean(h_errs):.2e}")
    print(f"  Forces: {np.mean(f_errs):.2e}")
    print(f"  Energy: {np.mean(e_errs):.2e}")

    print(f"Polar plots saved to {output_dir}")

if __name__ == "__main__":
    test_equivariance()