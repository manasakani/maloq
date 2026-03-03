"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import matplotlib.pyplot as plt # remove
import os, sys
import torch
import math
import torch.nn as nn
import e3nn
from e3nn.o3 import Irreps, Irrep
from e3nn.o3 import Linear as e3nn_Linear
from e3nn.nn import Gate
from torch.nn import Linear
import numpy as np
from abc import ABCMeta, abstractmethod
from torch.utils.checkpoint import checkpoint
import time
from .nn.so3_layers import SO3_Linear

import torch.distributed as dist
from mpi4py import MPI

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
e3nn.set_optimization_defaults(jit_script_fx=False)

from .common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
    eulers_to_wigner,
    init_edge_rot_euler_angles
)
from .common.so3 import (
    CoefficientMapping,
    SO3_Grid,
)
from .esen_block import eSEN_Block
from .nn.embedding import EdgeDegreeEmbedding
from .nn.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .nn.radial import EnvelopedBesselBasis, GaussianSmearing
from .nn.so2_layers import SO2_Convolution
from .nn.so3_layers import SO3_Linear
from .nn.activation import GateActivation

from .common.irreps_utils import get_reduced_to_all_indices, get_parity_multiplier

@registry.register_model("esen_backbone")
class eSEN_Backbone(nn.Module):
    def __init__(
        self,
        irreps_out,
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: int | None = None,
        max_neighbors: int = 300,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        cutoff = 10.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = False,
        regress_forces: bool = True,
        regress_stress: bool = False,
        # escnmd specific
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        mlp_type: str = "spectral",
        gaussian_width = 1.0,
        include_edges=True,
        open_shell=False,
        wigner_backend: str = "torch",
        distributed_graph_training=False
    ):
        super().__init__()

        assert wigner_backend in ("torch", "triton"), \
            f"wigner_backend must be 'torch' or 'triton', got '{wigner_backend}'"
        self.wigner_backend = wigner_backend
        self._wigner_buf = None  # upper-bound pre-allocated, grown only when num_edges exceeds capacity

        if not include_edges:
            print("Note: Initializing eSEN backbone without edge_embeddings!")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.comm = MPI.COMM_WORLD

        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.gaussian_width = gaussian_width

        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress
        self.mlp_type = mlp_type
        self.include_edges = include_edges      # whether to use embeddings for the edges as well

        if open_shell:                                      # output two sets of labels for alpha/beta fock
            self.num_spins = 2
        else:
            self.num_spins = 1

        # rotation utils
        Jd_list = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))
        for l in range(self.lmax + 1):
            self.register_buffer(f"Jd_{l}", Jd_list[l])
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        # lmax_lmax for node, lmax_mmax for edge
        self.SO3_grid = nn.ModuleDict()
        self.SO3_grid["lmax_lmax"] = SO3_Grid(
            self.lmax, self.lmax, resolution=grid_resolution, rescale=True
        )
        self.SO3_grid["lmax_mmax"] = SO3_Grid(
            self.lmax, self.mmax, resolution=grid_resolution, rescale=True
        )

        # atom embedding
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels
        )
        #  For charge: possible values are -10 to +10 (21 values)
        self.abs_max_charge = 10
        self.charge_embedding = nn.Embedding(
            2*self.abs_max_charge + 1, self.sphere_channels
        )
        self.max_spin_multiplicity = 11
        self.spin_embedding = nn.Embedding(
            self.max_spin_multiplicity, self.sphere_channels
        )

        # # These three will be combined together with a linear layer:
        self.scalar_node_embedding = nn.Linear(3 * self.sphere_channels, self.sphere_channels)

        # edge distance embedding
        self.cutoff = cutoff
        self.edge_channels = edge_channels
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                self.num_distance_basis,
                self.gaussian_width,
            )
        elif self.distance_function == "bessel":
            self.distance_expansion = EnvelopedBesselBasis(
                num_radial=self.num_distance_basis,
                cutoff=cutoff,
            )
            self.distance_expansion.offset = [self.cutoff]
            self.distance_expansion.num_output = self.num_distance_basis
        else:
            raise ValueError("Unknown distance function")

        # equivariant initial embedding
        # self.element_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels) # for antisym
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)

        # nn.init.uniform_(self.element_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)

        self.edge_channels_list = [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

        # TEMPORARY: edge degree embedding is currently not compatible with distributed graph training, skipping it for now!
        # if not distributed_graph_training:
        #     self.edge_degree_embedding = EdgeDegreeEmbedding(
        #             sphere_channels=self.sphere_channels,
        #             lmax=self.lmax,
        #             mmax=self.mmax,
        #             max_num_elements=self.max_num_elements,
        #             edge_channels_list=self.edge_channels_list,
        #             rescale_factor=5.0,
        #             cutoff=self.cutoff,
        #             mappingReduced=self.mappingReduced,
        #             out_mask=self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
        #                 self.lmax, self.mmax
        #             )
        #         )

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.act_type = act_type

        # Initialize the blocks for each layer
        self.node_blocks = nn.ModuleList()

        if self.include_edges:
            self.edge_blocks = nn.ModuleList()


        for _ in range(self.num_layers):
            node_block = eSEN_Block(
                self.sphere_channels,
                self.hidden_channels,
                self.lmax,
                self.mmax,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.cutoff,
                self.norm_type,
                self.act_type,
                self.mlp_type,
                self.include_edges,
                node_or_edge='node'
            )
            self.node_blocks.append(node_block)

            if self.include_edges:
                edge_block = eSEN_Block(
                    self.sphere_channels,
                    self.hidden_channels,
                    self.lmax,
                    self.mmax,
                    self.mappingReduced,
                    self.SO3_grid,
                    self.edge_channels_list,
                    self.cutoff,
                    self.norm_type,
                    self.act_type,
                    self.mlp_type,
                    self.include_edges,
                    node_or_edge='edge'
                )
                self.edge_blocks.append(edge_block)


        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels
        )

    def _get_rotmat_and_wigner(self, edge_distance_vecs):
        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_distance_vecs.dtype)
            for l in range(self.lmax + 1)
        ]

        if self.wigner_backend == "triton":
            from .triton_kernels import edge_vec_to_wigner_fused
            num_edges = edge_distance_vecs.shape[0]
            out_dim = (self.lmax + 1) ** 2
            if self._wigner_buf is None or num_edges > self._wigner_buf.shape[0]:
                self._wigner_buf = torch.zeros(
                    num_edges, out_dim, out_dim,
                    device=edge_distance_vecs.device,
                    dtype=torch.float32,
                )
            wigner = edge_vec_to_wigner_fused(
                edge_distance_vecs, Jd_buffers, lmax=self.lmax,
                out=self._wigner_buf[:num_edges],
            )
        else:
            euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
            wigner = eulers_to_wigner(
                euler_angles,
                0,
                self.lmax,
                Jd_buffers,
            )
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        return wigner, wigner_inv


    @conditional_grad(torch.enable_grad())
    def forward(self, batch, batch_index=None, output_dir=None):


        distributed_graph_training = batch.distributed_graph_training if "distributed_graph_training" in batch else False

        data_dict = {
            "pos": batch.pos,
            "edge_index": batch.edge_index.squeeze(0).reshape(2, -1),       # composed of local fraction of global node indices
            "edge_dist": batch.edge_attr,
            "nedges": len(batch.edge_index[0]),
            "natoms": len(batch.pos),
            "atomic_numbers": batch.atomic_numbers,                         # always global
            "charges": batch.charge,
            "spin_multiplicity": batch.spin_multiplicity,
            "num_atoms_in_molecule": batch.num_atoms_in_molecule if not distributed_graph_training else None
        }

        # The input edges are in xyz coordinates, we need to rotate them to the yzx coordinates expected by e3nn to be consistent with the data
        edge_distance_vec = data_dict["edge_dist"][:, [2, 3, 1]]
        edge_distance = data_dict["edge_dist"][:, 0]

        graph_dict = {
            "edge_index": data_dict["edge_index"],
            "edge_distance": edge_distance,
            "edge_distance_vec": edge_distance_vec,
            "partition": batch.fock_target_object[0].domain if distributed_graph_training else None
        }


        wigner, wigner_inv = self._get_rotmat_and_wigner(
            graph_dict["edge_distance_vec"]
        )


        # --> Rotation test:
        # rotated_edges_to_z_axis = torch.bmm(edge_rot_mat, graph_dict["edge_distance_vec"].unsqueeze(-1)).squeeze(-1)
        # rotated_edges_to_z_axis = torch.bmm(wigner[:, 1:4, 1:4], graph_dict["edge_distance_vec"].unsqueeze(-1)).squeeze(-1)
        # print("Rotated edges to z-axis: ", rotated_edges_to_z_axis) # only middle components should be nonzero (equal to distance)

        ###############################################################
        # Initialize node and edge embeddings
        ###############################################################


        # x_message: [data_dict["pos"].shape[0] = #nodes, self.sph_feature_size = (l_max+1)**2, self.sphere_channels = E]
        x_message_node = torch.zeros(
            data_dict["pos"].shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=data_dict["pos"].device,
            dtype=data_dict["pos"].dtype,
        )
        # set l = 0 components to the element embeddings + charge + spin:

        if distributed_graph_training:

            start_node = graph_dict['partition'].start_node
            end_node = graph_dict['partition'].end_node

            # dist.barrier()
            # print(f"Rank {self.rank} processing nodes from {start_node} to {end_node}", flush=True)
            # print(f"Rank {self.rank} atomic numbers: {data_dict['atomic_numbers']}", flush=True)
            # dist.barrier()

            atom_charges = data_dict["charges"] + self.abs_max_charge
            element_emb = self.sphere_embedding(data_dict["atomic_numbers"][start_node : end_node])
            charge_emb = self.charge_embedding(atom_charges)
            spin_emb = self.spin_embedding(data_dict["spin_multiplicity"])

        else:

            # Seperate batch nodes into their molecules
            molecule_indices = torch.cat([
                torch.full((data_dict['natoms'],), i, dtype=torch.long, device=data_dict["pos"].device)
                for i, data_dict['natoms'] in enumerate(data_dict["num_atoms_in_molecule"])
            ])
            atom_charges = data_dict["charges"][molecule_indices] + self.abs_max_charge         # shape: [total_num_atoms]
            atom_spins = data_dict["spin_multiplicity"][molecule_indices]                       # shape: [total_num_atoms]

            element_emb = self.sphere_embedding(data_dict["atomic_numbers"])
            charge_emb = self.charge_embedding(atom_charges)
            spin_emb = self.spin_embedding(atom_spins)

        # x_message_node[:, :, 0, :] = element_emb + charge_emb + spin_emb # dims: [spin, nodes, l, E]

        # dist.barrier()
        # print(f"Rank {self.rank} shape of element_emb: {element_emb.shape}, charge_emb: {charge_emb.shape}, spin_emb: {spin_emb.shape}", flush=True)
        # dist.barrier()

        # Concatenate along the last dimension
        combined_emb = torch.cat([element_emb, charge_emb, spin_emb], dim=-1) # [num_atoms, 3 * sphere_channels]
        final_emb = self.scalar_node_embedding(combined_emb)                  # [num_atoms, sphere_channels]
        x_message_node[:, 0, :] = final_emb

        if self.include_edges:
            # x_message_edge: [#edges considered, self.sph_feature_size = (l_max+1)**2, self.sphere_channels = E]
            x_message_edge = torch.zeros(
                data_dict["nedges"],
                self.sph_feature_size,
                self.num_distance_basis, #self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            # set l = 0 components to the distance expansion
            x_message_edge[:, 0, :] = self.distance_expansion(graph_dict["edge_distance"]) # maybe remove
        else:
            x_message_edge = None

        # edge embedding: [num_edges, num gaussian basis functions]
        # source_embedding, target_embedding: [num_edges, self.sphere_channels]
        # x_edge: [num_edges, num gaussian basis functions + 2*self.sphere_channels]


        edge_distance_embedding = self.distance_expansion(graph_dict["edge_distance"])

        source_embedding = self.source_embedding(
            data_dict["atomic_numbers"][graph_dict["edge_index"][0]]
        )

        target_embedding = self.target_embedding(
            data_dict["atomic_numbers"][graph_dict["edge_index"][1]]
        )

        x_edge = torch.cat((source_embedding, edge_distance_embedding, target_embedding), dim=1) 


        # do edge degree embeddings for both nodes and edges:
        # need to redo src/target nn mapping in the distributed graph before using this!!
        # if graph_dict["partition"] is None:
        #     x_message_node = self.edge_degree_embedding( 
        #         x_message_node,
        #         x_edge,
        #         graph_dict["edge_distance"],
        #         graph_dict["edge_index"],
        #         wigner_inv,
        #         node_or_edge='node',
        #         partition=graph_dict["partition"]
        #     )

        #     if self.include_edges:
        #         x_message_edge = self.edge_degree_embedding(
        #             x_message_node,
        #             x_edge,
        #             graph_dict["edge_distance"],
        #             graph_dict["edge_index"],
        #             wigner_inv,
        #             node_or_edge='edge',
        #             partition=None
        #         )
        # else:
        #     print("Warning: edge degree embedding is currently not compatible with distributed graph training, skipping it for now!", flush=True)

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):

            x_message_node  = self.node_blocks[i](
                x_message_node,
                x_message_edge,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner,
                wigner_inv,
                node_or_edge='node',
                partition=graph_dict["partition"]
            )

            if self.include_edges:
                x_message_edge = self.edge_blocks[i](
                    x_message_node,
                    x_message_edge,
                    x_edge,
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    wigner,
                    wigner_inv,
                    node_or_edge='edge',
                    partition=graph_dict["partition"]
                )
            

        # Final layer norm
        x_message_node = self.norm(x_message_node)

        if self.include_edges:
            x_message_edge  = self.norm(x_message_edge)

        # Return the output
        if self.include_edges: # all we need for the fock output head
            out = {
                    "node_embeddings": x_message_node,
                    "edge_embeddings": x_message_edge,
                }
        else:
            print("Error: figure out how to handle the spin dimension in the energy head before using this!")
            exit()
            out = {
                    "node_embeddings": x_message_node,
                    "edge_embeddings": x_message_edge,
                    "x_edge": x_edge,
                    "wigner": wigner,
                    "wigner_inv": wigner_inv
                }
        out.update(graph_dict)
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_Linear,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)



@registry.register_model("fock_irreps_head")
class Fock_Irreps_Head(nn.Module):
    """
    Takes an input irrep like 64x0e+64x1e+64x2e ... and nonlinearly maps it to the output irreps of arbitrary size and l-multiplicity
    """
    def __init__(self, irreps_in,
                irreps_out,
                lmax,
                sphere_channels,
                head_type='gated',
                half_edges=False,
                ls_list=None,
                open_shell=False,
                reduce_node=False,
                reduce_node_intra=False,
                orbital_basis=None):

        super().__init__()

        self.head_type = head_type
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.reduce_node = reduce_node                      # take advantage of 'inter'-orbital interaction symmetry within node blocks
        self.reduce_node_intra = reduce_node_intra          # take advantage of 'intra'-orbital interaction symmetry within node blocks
        self.irreps_out = irreps_out
        self.half_edges = half_edges                        # if only half of the edges were created in the database to compute the loss over
        self.ls_list = ls_list                              # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        self.backward_irrep_track = {}                      # helper dict to keep track of where to find the forward onsite orbital edges when we expand them out later
        print("ls_list for output head: ", self.ls_list)

        if open_shell:                                      # output two sets of labels for alpha/beta fock
            self.num_spins = 2
        else:
            self.num_spins = 1

        # We make a new list of irreps (irreps_nodereduced) which contains only the unique irreps in the node blocks
        if self.reduce_node:

            self.reduced_to_all_indices = get_reduced_to_all_indices(self.ls_list, reduce_node_intra=self.reduce_node_intra)
            parity_multiplier = torch.asarray(get_parity_multiplier(self.ls_list, reduce_node_intra=self.reduce_node_intra), dtype=torch.float32)

            # register buffer for parity multiplier so it can be used in the forward pass and is on the correct device/dtype
            self.register_buffer("parity_multiplier", parity_multiplier)

            irreps_nodereduced = []
            irrep_pointer = 0
            for i, l1 in enumerate(self.ls_list):
                for j, l2 in enumerate(self.ls_list):

                    # if this is an orbital self-interaction within the node block, we add the even irreps
                    if i == j and l1 == l2:
                        if self.reduce_node_intra:
                            product_irreps = str(self.get_product_irreps(l1, l2, 'even'))
                        else:
                            product_irreps = str(self.get_product_irreps(l1, l2))

                        irreps_nodereduced.append(product_irreps)
                        irrep_pointer += sum([2*l + 1 for l in Irreps(product_irreps).ls])

                    # this is an upper-triangle off-diag interaction within the node block, we add all the required irreps
                    if i < j:
                        product_irreps = str(self.get_product_irreps(l1, l2))
                        irreps_nodereduced.append(product_irreps)
                        irrep_len = sum([2*l + 1 for l in Irreps(product_irreps).ls])

                        # track it for the backward edge in the expand section later:
                        self.backward_irrep_track[(j, i)] = [irrep_pointer, irrep_pointer+irrep_len]
                        irrep_pointer += irrep_len

            # Now we can project to this reduced set of irreps, and expand it out later
            self.irreps_nodereduced = Irreps('+'.join(irreps_nodereduced))

        # This permutation list and reflection vector together define the relationship between the forward and backward edges
        if self.half_edges:
            assert self.ls_list is not None, "If you use half edges, you need to provide the ls_list!"
            self.edge_m_reflection = None
            self.edge_permutation = self.get_edge_permutation()

        irreps_scalars, irreps_gated = self.split_irreps(irreps_in)
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

        # 1. Apply a linear layer to convert the number of input scalars to the number of required gating scalars
        # the number of input scalars is equal to sphere_channels
        # the output 'irreps_gates' are the gating scalars

        # --> gate with learnable parameters by outputting more random scalars:
        input_scalars_irreps = Irreps(f"{self.sphere_channels}x0e")
        combined_output_scalars = Irreps(f"{irreps_scalars.num_irreps + irreps_gated.num_irreps}x0e")
        self.lin_scalars_learnable = nn.ModuleList()
        self.act_input_scalars = nn.ModuleList()
        self.gate = nn.ModuleList()

        if reduce_node:
            self.node_lin_out_layers = nn.ModuleList()
            self.edge_lin_out_layers = nn.ModuleList()
            self.node_output_permutation = self.get_output_permutation(self.irreps_nodereduced)
            self.edge_output_permutation = self.get_output_permutation(irreps_out)
        else:
            self.lin_out_layers = nn.ModuleList()
            self.output_permutation = self.get_output_permutation(irreps_out)

        for spin in range(self.num_spins):

            self.lin_scalars_learnable.append(e3nn_Linear(irreps_in=input_scalars_irreps, irreps_out=combined_output_scalars, biases=True))
            self.act_input_scalars.append(torch.nn.SiLU())
            # this returns the irreps_gates

            # 2. Apply the gating to the other ls (need to pass in a stack of [l=0, l~=0])
            self.gate.append(Gate(irreps_scalars=Irreps(),
                                    act_scalars=[],
                                    irreps_gates=irreps_gates,
                                    act_gates=[torch.sigmoid] * len(irreps_gates),
                                    irreps_gated=irreps_gated
                                ))

            # now we have the [l=0s, gated l>0s] in a stack, and we just need to map them to the output irrep order:
            irreps_in_simplified = (irreps_scalars + self.gate[spin].irreps_out).simplify()
            assert irreps_in_simplified == (irreps_scalars+self.gate[spin].irreps_out), "The irreps_in for the output linear layer should not change when simplified!"
            if reduce_node:
                node_lin_layers = nn.ModuleList()
                edge_lin_layers = nn.ModuleList()
                for l in range(0, self.lmax+1):
                    mul_in = self.sphere_channels

                    # for nodes:
                    mul_out = self.irreps_nodereduced.count('{}e'.format(l))
                    irreps_in_l = f"{mul_in}x{l}e"
                    irreps_out_l = f"{mul_out}x{l}e"
                    print("Creating node linear output map for l = ", l, " with irreps_in = ", irreps_in_l, " and irreps_out = ", irreps_out_l, flush=True)
                    node_lin_layers.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l, biases=True))

                    # for edges:
                    mul_out = self.irreps_out.count('{}e'.format(l))
                    irreps_in_l = f"{mul_in}x{l}e"
                    irreps_out_l = f"{mul_out}x{l}e"
                    print("Creating edge linear output map for l = ", l, " with irreps_in = ", irreps_in_l, " and irreps_out = ", irreps_out_l, flush=True)
                    edge_lin_layers.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l, biases=True))

                self.node_lin_out_layers.append(node_lin_layers)
                self.edge_lin_out_layers.append(edge_lin_layers)
            else:

                # create single linear layers up to lmax:
                lin_layers = nn.ModuleList()
                for l in range(0, self.lmax+1):
                    mul_in = self.sphere_channels
                    mul_out = irreps_out.count('{}e'.format(l))
                    irreps_in_l = f"{mul_in}x{l}e"
                    irreps_out_l = f"{mul_out}x{l}e"
                    print("Creating linear output map for l = ", l, " with irreps_in = ", irreps_in_l, " and irreps_out = ", irreps_out_l, flush=True)
                    lin_layers.append(e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l, biases=True))
                    # layers = nn.Sequential( # more weights
                    #     e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_in_l, biases=True),
                    #     e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_in_l, biases=True),
                    #     e3nn_Linear(irreps_in=irreps_in_l, irreps_out=irreps_out_l, biases=True)
                    # )
                    # lin_layers.append(layers)

                self.lin_out_layers.append(lin_layers)

    def get_output_permutation(self, output_irreps):
        """
        Get the permutation that reorders irreps from sorted-by-l order to irreps_out order.
        Initially, we have the output irreps in sorted order: (omol) 114x0e+ 229x1e+239x2e+161x3e+73x4e+24x5e+4x6e
        We want to permute them to the order they appear in irreps_out.
        """
        input_irreps_simplified = output_irreps.sort()[0].simplify()
        total_dim = sum(mul * (2 * ir.l + 1) for mul, ir in input_irreps_simplified)
        # print("Total output dim: ", total_dim, flush=True)

        sorted_irreps, permutation, inverse_permutation = output_irreps.sort()
        # Inverse permutation: output irreps -> sorted irreps
        # Permutation: sorted irreps -> output irreps

        req_permutation = list(inverse_permutation)

        # augment each element of permutation by the total dimension of all previous irreps:
        tensor_inverse_permutation = torch.zeros(total_dim, dtype=torch.long)
        sorted_tensor_pos = 0

        # For each irrep in sorted order, find where it should go in original order
        for sorted_irrep_idx, (_, ir) in enumerate(sorted_irreps):
            ir_dim = 2 * ir.l + 1
            # print("Handling irrep: ", ir, " of dim ", ir_dim, "in sorted position ", sorted_irrep_idx, flush=True)
            # print("This should go to unsorted position ", req_permutation[sorted_irrep_idx], flush=True)

            # Find which original irrep this corresponds to in the output_irreps
            original_irrep_idx = req_permutation[sorted_irrep_idx]

            # Calculate tensor position in original order for this irrep
            original_tensor_pos = 0
            for i in range(original_irrep_idx):
                _, orig_ir = output_irreps[i]
                original_tensor_pos += 2 * orig_ir.l + 1

            # Map tensor elements: original[orig_pos] = sorted[sorted_pos]
            for j in range(ir_dim):
                tensor_inverse_permutation[original_tensor_pos + j] = sorted_tensor_pos + j

            sorted_tensor_pos += ir_dim

        # print("tensor_inverse_permutation: ", [tensor_inverse_permutation[i].item() for i in range(len(tensor_inverse_permutation))], flush=True)
        return tensor_inverse_permutation


    def split_irreps(self, irreps):
        scalars = []
        gated = []
        for mul, ir in irreps:
            if ir.l == 0:
                scalars.append((mul, ir))
            else:
                gated.append((mul, ir))
        return Irreps(scalars), Irreps(gated)


    def get_product_irreps(self, l1, l2, even_or_odd=None):
        """
        Return the irreps required to represent l1 X l2 (X = tensor product)
        """

        m = 1   # multiplicity
        p = 1   # even parity only (real-valued Fock matrix)
        l3s = range(abs(l1 - l2), l1 + l2 + 1)

        # return only the even/odd irreps:
        if even_or_odd is not None:
            if even_or_odd == 'even':
                even_l3s = [l for l in l3s if l % 2 == 0]
                required_irreps = Irreps([(m, (l, p)) for l in even_l3s])
            else:
                odd_l3s = [l for l in l3s if l % 2 != 0]
                required_irreps = Irreps([(m, (l, p)) for l in odd_l3s])
        else:
            required_irreps = Irreps([(m, (l, p)) for l in l3s])

        return required_irreps


    def forward(self, emb, batch):

        node_embeddings = emb["node_embeddings"]
        edge_embeddings = emb["edge_embeddings"]
        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)

        edge_mask = batch.edge_mask if "edge_mask" in batch else None

        node_outputs = []
        edge_outputs = []
        for spin in range(self.num_spins):
            node_embeddings_spin = self.stack_irreps(node_embeddings)
            edge_embeddings_spin = self.stack_irreps(edge_embeddings)
            node_output = self.process(node_embeddings_spin, edge_index, 'node', spin)
            edge_output = self.process(edge_embeddings_spin, edge_index, 'edge', spin)

            # augment the node irreps back to the full irrep list (containing the lower triangle of orbital interactions and odd self-interaction irreps)
            # using edge_output to infer the total size of the output embeddings
            if self.reduce_node:
                node_output = self.expand_reduced_node(node_output, edge_output)
                        
            node_outputs.append(node_output)
            edge_outputs.append(edge_output)

        # Stack along spin dimension
        node_outputs = torch.stack(node_outputs, dim=0)  # [spin, num_nodes, ...]
        edge_outputs = torch.stack(edge_outputs, dim=0)  # [spin, num_edges, ...]
        return node_outputs, edge_outputs

    def get_edge_permutation(self):
        """
        The forward and backward edges contain the same irreps, but they are permuted in the data list
        due to the order of flattening the matrix blocks. Here we create the permutation of the irreps to match the reverse edge order.
        We also handle the reflection rules of the orbital interactions, which are different for even and odd parity.
        """

        full_irrep_len = [sum([2*l + 1 for l in Irreps(str(self.get_product_irreps(l1, l2))).ls]) for l1 in self.ls_list for l2 in self.ls_list]
        edge_permutation = [0] * sum(full_irrep_len)
        self.edge_m_reflection = np.ones(sum(full_irrep_len), dtype=int)
        forward_irrep_track = {}
        pointer = 0

        total_irreps = Irreps('')

        for i, l1 in enumerate(self.ls_list):
            for j, l2 in enumerate(self.ls_list):

                # --> 1. Handle the permutation of the irreps:
                product_irreps = str(self.get_product_irreps(l1, l2))
                irrep_len = sum([2*l + 1 for l in Irreps(product_irreps).ls])

                # add to total irreps
                total_irreps += Irreps(product_irreps)

                # if it's the same orbital interaction going backward and forward (eg, p1A-p1B vs. p1B-p1A), we keep the same irreps
                if i == j:
                    edge_permutation[pointer:pointer+irrep_len] = [pointer + i for i in range(irrep_len)]

                # if its an interaction between different orbitals (eg, p1A-p2B vs. p2B-p1A), we append the index of the permutation
                if i < j:
                    # store this in the forward_irrep_track:
                    forward_irrep_track[(j, i)] = [pointer, pointer + irrep_len]

                if i > j:

                    # Find where the p1A-p2B irreps are in the forward edge
                    forward_irrep_start = forward_irrep_track[(i, j)][0]
                    forward_irrep_end = forward_irrep_track[(i, j)][1]

                    # Update both the forward and backward edge permutations
                    edge_permutation[pointer:pointer+irrep_len] = list(range(forward_irrep_start, forward_irrep_end))
                    edge_permutation[forward_irrep_start:forward_irrep_end] = list(range(pointer, pointer + irrep_len))

                # --> 2. Handle the reflections
                parity = ((-1) ** (l1+l2)).item()

                # Even parity: odd output irreps are flipped
                if parity == 1:
                    start_l = 0
                    for p in product_irreps.split('+'):
                        l = Irreps(p).ls[0]
                        if l % 2 != 0:
                            l_orb_start = pointer + start_l
                            l_orb_end = l_orb_start + (2*l + 1)
                            self.edge_m_reflection[l_orb_start:l_orb_end] *= -1
                        start_l += (2*l + 1)

                # Odd parity: even output irreps are flipped
                if parity == -1:
                    start_l = 0
                    for p in product_irreps.split('+'):
                        l = Irreps(p).ls[0]
                        if l % 2 == 0:
                            l_orb_start = pointer + start_l
                            l_orb_end = l_orb_start + (2*l + 1)
                            self.edge_m_reflection[l_orb_start:l_orb_end] *= -1
                        start_l += (2*l + 1)

                pointer += irrep_len

        # assert that total_irreps is the same as the output irreps
        assert total_irreps == self.irreps_out, f"Error! Total irreps in the Hamiltonian output head {total_irreps} do not match the provided output irreps {self.irreps_out}!"

        return edge_permutation


    def stack_irreps(self, x_message):
        # input = x_message = [num_atoms/edges (batch_size), (lmax+1)**2, sphere_channels]

        x_message_T = x_message.transpose(-1,-2) # rearrange dimensions from [l, E] to [E, l]
        batch_size = x_message_T.shape[0]

        # group all the different ls so l_sorted output looks like sphere_channels*0e + sphere_channels*1e + sphere_channels*2e ...
        l_sorted_output = torch.zeros(batch_size, self.sphere_channels*((self.lmax+1)**2), device=x_message.device)
        for l in range(self.lmax+1):
            start = (l**2)*self.sphere_channels
            end = (l**2)*self.sphere_channels + self.sphere_channels*(2*l+1)
            l_sorted_output[:,start:end] = torch.squeeze(x_message_T[:, :, (l**2):(l**2)+(2*l+1)].reshape(batch_size, 1, -1))

        return l_sorted_output

    def process(self, x, edge_index, node_or_edge, spin):

        # 1. Extract the scalar components, which are the first # sphere_channels elements of this tensor
        x_scalars = x[:, :self.sphere_channels]
        x_nonscalars = x[:, self.sphere_channels:]

        # 2. Prepare some scalars for gating - gate with learnable scalars: the first 'sphere_channels' scalars are the l=0, and others are used for gating
        all_scalars = self.lin_scalars_learnable[spin](x_scalars)
        transformed_l0_scalars = all_scalars[:, :self.sphere_channels]
        gating_scalars = all_scalars[:, self.sphere_channels:]

        # 3. Gate the l>0 irreps:
        x_gated = self.gate[spin](torch.cat([gating_scalars, x_nonscalars], dim=1))
        x_gated = torch.cat([transformed_l0_scalars, x_gated], dim=1)   # use the transformed scalars as the output

        # 4. Apply linear map to output irreps
        if not self.reduce_node:
            # lin_out_start_time = time.perf_counter()

            # pass each irrep of x_out through its own SO3_Linear layer, where x_out followed simplified irreps_in (self.sphere_channels*0e + sphere_channels*1e + sphere_channels*2e ...)
            x_out_list = []
            irrep_end_track = 0
            batch_size = x_gated.shape[0]
            for l in range(0, self.lmax+1):

                # sphere channels is the multiplicity of this l in the input
                start_idx = irrep_end_track
                end_idx = start_idx + self.sphere_channels  * (2*l + 1)
                irrep_end_track = end_idx

                x_l = x_gated[:, start_idx:end_idx]  # extract the l-th irrep component

                # x_l = x_l.reshape(batch_size, 2*l + 1, self.sphere_channels)  # this is for SO3_Linear..
                # if node_or_edge == 'node':
                #     x_l_out = self.node_lin_out_layers[l](x_l)  # apply the Linear layer for degree l
                # if node_or_edge == 'edge':
                #     x_l_out = self.edge_lin_out_layers[l](x_l)  # apply the Linear layer for degree l
                x_l_out = self.lin_out_layers[spin][l](x_l)  # apply the Linear layer for degree l
                # mul_out = self.irreps_out.count(f'{l}e')
                # x_l_out = x_l_out.reshape(batch_size, mul_out*(2*l+1))  # [batch, (2*l+1)*mul_out]

                x_out_list.append(x_l_out)

            # concatenate all the l outputs back together
            x_out = torch.cat(x_out_list, dim=1)

            # permute to match the expected order of irreps in the output
            x_out = x_out[:, self.output_permutation]

            # lin_out_end_time = time.perf_counter()
            # print(f"lin_out time: {lin_out_end_time - lin_out_start_time} seconds", flush=True)

        else:
            x_out_list = []
            irrep_end_track = 0
            batch_size = x_gated.shape[0]
            for l in range(0, self.lmax+1):

                # sphere channels is the multiplicity of this l in the input
                start_idx = irrep_end_track
                end_idx = start_idx + self.sphere_channels  * (2*l + 1)
                irrep_end_track = end_idx

                x_l = x_gated[:, start_idx:end_idx]  # extract the l-th irrep component
                if node_or_edge == 'node':
                    x_l_out = self.node_lin_out_layers[spin][l](x_l)  # apply the Linear layer for degree l
                if node_or_edge == 'edge':
                    x_l_out = self.edge_lin_out_layers[spin][l](x_l)  # apply the Linear layer for degree l
                x_out_list.append(x_l_out)

            # concatenate all the l outputs back together
            x_out = torch.cat(x_out_list, dim=1)

            # permute to match the expected order of irreps in the output
            if node_or_edge == 'node':
                x_out = x_out[:, self.node_output_permutation]
            if node_or_edge == 'edge':
                x_out = x_out[:, self.edge_output_permutation]

        return x_out


    def expand_reduced_node(self, node_output, edge_output):
        """
        Expand irreps_nodereduced to irreps_out, by adding the previously-removed irreps back in:
        1. The odd irreps for the orbital self-interactions
        2. The 'lower triangle' of inter-orbital interactions on this node (eg the p-s to s-p)
        """

        expanded_node_output = node_output[:, self.reduced_to_all_indices] * self.parity_multiplier
        return expanded_node_output


@registry.register_model("esen_linear_energy_head")
class Linear_Energy_Head(nn.Module):
    def __init__(self, backbone, include_edges=True):
        super().__init__()

        self.include_edges = include_edges
        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.lmax = backbone.lmax
        self.mmax = backbone.mmax
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)  # need to re-create mapping reduced to avoid inplace modification error!
        self.edge_channels_list = backbone.edge_channels_list
        extra_m0_output_channels = self.lmax * self.hidden_channels

        # Convolution method:
        self.act = GateActivation(
                lmax=self.lmax, mmax=self.mmax, num_channels=self.hidden_channels
            )

        multiplier = 2 #3 if self.include_edges else 2

        self.so2_conv_1 = SO2_Convolution(
            multiplier*self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
        )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.sphere_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

        # self.linear = nn.Linear(backbone.sphere_channels, 1, bias=True)

        # make this a two linear layers with nonlinearity in between:
        # self.linear = nn.Sequential(
        #     nn.Linear(self.sphere_channels, 2*self.sphere_channels, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(2*self.sphere_channels, 2*self.sphere_channels, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(2*self.sphere_channels, 1, bias=True),
        # )
        h = 2*self.sphere_channels
        self.linear = nn.Sequential(
            nn.Linear(self.sphere_channels, h, bias=True),
            nn.SiLU(),
            nn.Linear(h, 1, bias=True),
        )


    def forward(self, emb: dict[str, torch.Tensor], batch):

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)
        # edge_mask = batch.edge_mask
        edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

        # Trim the embeddings to the chosen lmax (not used)
        nodes = emb["node_embeddings"]#[:, :(self.lmax+1)**2, :]
        edges = emb["edge_embeddings"]#[:, :(self.lmax+1)**2, :]
        x_edge = emb["x_edge"]
        wigner = emb["wigner"]#[:, :(self.lmax+1)**2, :(self.lmax+1)**2]
        wigner_inv = emb["wigner_inv"]#[:, :(self.lmax+1)**2, :(self.lmax+1)**2]

        # Create the messages for the last convolution:
        x_source = nodes[edge_index[0][edge_mask]]
        x_target = nodes[edge_index[1][edge_mask]]
        x_message = torch.cat((x_source, x_target), dim=2)

        # -----------------
        # Rotate the irreps
        x_message = torch.bmm(wigner, x_message)

        # Apply the SO2 convolution to the messages
        x_message, x_0_gating = self.so2_conv_1(x_message, x_edge)
        x_message = self.act(x_0_gating, x_message)
        x_message = self.so2_conv_2(x_message, x_edge)

        # Rotate back the irreps
        x_message = torch.bmm(wigner_inv, x_message)

        # Compute the sum of the incoming neighboring messages for each target node
        new_embedding = torch.zeros(
            (nodes.shape[0],) + x_message.shape[1:],
            dtype=x_message.dtype,
            device=x_message.device,
        )

        # aggregate messages
        new_embedding.index_add_(0, edge_index[1][edge_mask], x_message) # only for the first row
        energies = new_embedding.narrow(1, 0, 1)
        energies = self.linear(energies)

        energies = energies.squeeze(-1).squeeze(-1)

        molecule_indices = torch.cat([
            torch.full((num_atoms,), i, dtype=torch.long, device=energies.device)
            for i, num_atoms in enumerate(batch.num_atoms_in_molecule)
        ])

        # atom-resolved energy -> molecule-resolved energy
        num_molecules = batch.num_atoms_in_molecule.size(0)
        energies_per_molecule = torch.zeros(num_molecules, device=energies.device)
        energies_per_molecule = energies_per_molecule.scatter_add(0, molecule_indices, energies)

        return {"energies": energies_per_molecule}


@registry.register_model("esen_linear_force_head")
class Linear_Force_Head(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, emb: dict[str, torch.Tensor], batch):

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)

        forces = self.linear(emb["node_embeddings"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        return {"forces": forces}