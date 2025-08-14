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
from e3nn.o3 import Irreps, Irrep 
from e3nn.o3 import Linear as e3nn_Linear
from e3nn.nn import Gate
from torch.nn import Linear
import numpy as np
from abc import ABCMeta, abstractmethod

# Fix this later!:
# sys.path.append('/home/manasakani/fairchem/src/')
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad

from .common.rotation import (
    init_edge_rot_mat,
    rotation_to_wigner,
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
        include_edges=True
    ):
        super().__init__()

        if not include_edges:
            print("Note: Initializing eSEN backbone without edge_embeddings!")

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

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            cutoff=self.cutoff,
            mappingReduced=self.mappingReduced,
            out_mask=self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(
                self.lmax, self.mmax
            )
        )

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

        # affine=True, centering=True # for antisymmetry, removes the bias!

        # # Irreps of the internal embeddings
        # build_irreps = []
        # for l in range(self.lmax+1):
        #     build_irreps.append((self.sphere_channels, (l, 1)))
        # self.irreps_in = Irreps(build_irreps)

    def get_rotmat_and_wigner(self, edge_distance_vecs):

        edge_rot_mat = init_edge_rot_mat(
            edge_distance_vecs, rot_clip=(not self.direct_forces)
        )

        Jd_buffers = [
            getattr(self, f"Jd_{l}").type(edge_rot_mat.dtype)
            for l in range(self.lmax + 1)
        ]

        wigner = rotation_to_wigner(
            edge_rot_mat,
            0,
            self.lmax,
            Jd_buffers,
            rot_clip=(not self.direct_forces),
        )
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        return edge_rot_mat, wigner, wigner_inv


    @conditional_grad(torch.enable_grad())
    def forward(self, batch, batch_index=None, output_dir=None):

        data_dict = {
            "pos": batch.pos,
            "edge_index": batch.edge_index.squeeze(0).reshape(2, -1),
            "forward_edge_mask": batch.edge_mask if "edge_mask" in batch else None,
            "reverse_edge_map": batch.reverse_edge_map if "reverse_edge_map" in batch else None,
            "edge_dist": batch.edge_attr,
            "nedges": len(batch.edge_index[0]),
            "natoms": len(batch.pos),
            "atomic_numbers": batch.atomic_numbers
        }
        # collect forward edges:
        forward_edge_mask = data_dict["forward_edge_mask"]  # which edges in edge_index are 'forward', and computed explicitly. eg: [T, T, T, F, F, F] for H2O with [[0,0,1,1,2,2], [1,2,2,0,0,1]]
        reverse_edge_map = data_dict["reverse_edge_map"]    # the elements corresponding to T in forward_edge_mask have their own index, and the ones corresponding to F have the index of their forward edge
                                                            # for the H2o example, reverse_edge_map = [0, 1, 2, 0, 1, 2] because the last three are the backward edges of the first three

        # In case these are not provided, we use all the edges
        if forward_edge_mask is None:
            forward_edge_mask = torch.ones(data_dict["edge_index"].shape[1], dtype=torch.bool, device=data_dict["pos"].device)
            reverse_edge_map = torch.arange(data_dict["edge_index"].shape[1], dtype=torch.long, device=data_dict["pos"].device)

        # The input edges are in xyz coordinates, we need to rotate them to the yzx coordinates expected by e3nn to be consistent with the data       
        edge_distance_vec = data_dict["edge_dist"][:, [2, 3, 1]][forward_edge_mask]  
        edge_distance = data_dict["edge_dist"][:, 0][forward_edge_mask] 
        
        graph_dict = {
            "edge_index": data_dict["edge_index"],  # this is the full edge_index, forward and backward
            "forward_edge_mask": forward_edge_mask,
            "reverse_edge_map": reverse_edge_map,
            "edge_distance": edge_distance,         # corresponds to only the masked edges
            "edge_distance_vec": edge_distance_vec,
        }

        edge_rot_mat, wigner, wigner_inv = self.get_rotmat_and_wigner(
            graph_dict["edge_distance_vec"]
        )

        forward_edges = None

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
        # set l = 0 components to the element embeddings:
        x_message_node[:, 0, :] = self.sphere_embedding(data_dict["atomic_numbers"]) 

        if self.include_edges:
            # x_message_edge: [#edges considered, self.sph_feature_size = (l_max+1)**2, self.sphere_channels = E]
            x_message_edge = torch.zeros(
                forward_edge_mask.sum().item(), 
                self.sph_feature_size,
                self.num_distance_basis, #self.sphere_channels,
                device=data_dict["pos"].device,
                dtype=data_dict["pos"].dtype,
            )
            # set l = 0 components to the distance expansion
            x_message_edge[:, 0, :] = self.distance_expansion(graph_dict["edge_distance"])
        else:
            x_message_edge = None

        # edge embedding: [num_edges, num gaussian basis functions]
        # source_embedding, target_embedding: [num_edges, self.sphere_channels]
        # x_edge: [num_edges, num gaussian basis functions + 2*self.sphere_channels]

        edge_distance_embedding = self.distance_expansion(graph_dict["edge_distance"])

        source_embedding = self.source_embedding(
            data_dict["atomic_numbers"][graph_dict["edge_index"][0]][forward_edge_mask]
        )

        target_embedding = self.target_embedding(
            data_dict["atomic_numbers"][graph_dict["edge_index"][1]][forward_edge_mask]
        )

        # x_edge needs to be symmetric over edges: (testing with sym)
        x_edge = torch.cat((source_embedding, edge_distance_embedding, target_embedding), dim=1) #+ torch.cat((target_embedding, edge_distance_embedding, source_embedding), dim=1)      # symmetrized

        # do edge degree embeddings for both nodes and edges: - this breaks symmetry of identical nodes..
        x_message_node = self.edge_degree_embedding(
            x_message_node,
            x_edge,
            graph_dict["edge_distance"],
            graph_dict["edge_index"],
            graph_dict["forward_edge_mask"],
            wigner_inv,
            node_or_edge='node'
        )

        if self.include_edges: # this is not antisymmetrized (yet) 
            x_message_edge = self.edge_degree_embedding(
                x_message_node,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                graph_dict["forward_edge_mask"],
                wigner_inv,
                node_or_edge='edge'
            )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            x_message_node = self.node_blocks[i](
                x_message_node,
                x_message_edge,
                x_edge,
                forward_edges,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                graph_dict["forward_edge_mask"],
                graph_dict["reverse_edge_map"],
                wigner,
                wigner_inv,
                node_or_edge='node',
            )

            if self.include_edges:
                x_message_edge = self.edge_blocks[i](
                    x_message_node,
                    x_message_edge,
                    x_edge,
                    forward_edges,
                    graph_dict["edge_distance"],
                    graph_dict["edge_index"],
                    graph_dict["forward_edge_mask"],
                    graph_dict["reverse_edge_map"],
                    wigner,
                    wigner_inv,
                    node_or_edge='edge',
                )
        
        # Final layer norm
        x_message_node = self.norm(x_message_node)

        if self.include_edges:
            x_message_edge = self.norm(x_message_edge)

        # Return the output
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
    def __init__(self, irreps_in, irreps_out, lmax, sphere_channels, head_type='gated', reduce_node=False, reduce_node_intra=False, orbital_basis=None):
        super().__init__()

        self.head_type = head_type
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.reduce_node = reduce_node                      # take advantage of 'inter'-orbital interaction symmetry within node blocks
        self.reduce_node_intra = reduce_node_intra          # take advantage of 'intra'-orbital interaction symmetry within node blocks
        self.irreps_out = irreps_out
        self.orbital_basis = orbital_basis
        
        # --> Option to extract minimal node irreps to project to:
        # NOTE: only use this explicitly in the forward pass, where the odd components are manually filtered out to produce a symmetric matrix
        # it seems to be useful to keep the zeros in during training, so the network learns that those components should be zero.
        # if self.reduce_node:
        assert self.orbital_basis is not None
        ls_list = []
        N = 0
        for l in range(5): # searching for up to g orbitals
            counts = [torch.sum(self.orbital_basis[el] == l) for el in self.orbital_basis]
            ls_list.append(torch.tensor(max(counts) * [l], dtype=torch.int))

        self.ls_list = torch.cat(ls_list)        # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        self.backward_irrep_track = {}           # helper dict to keep track of where to find the forward edges when we expand them out later

        # We make a new list of irreps (irreps_nodereduced) which contains only the unique irreps in the node blocks
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
        self.edge_m_reflection = None
        self.edge_permutation = self.get_edge_permutation()

        if self.head_type == 'linear':
            self.map_node_to_rank_N = e3nn_Linear(irreps_in=irreps_in, irreps_out=irreps_out, biases=True)
            self.map_edge_to_rank_N = e3nn_Linear(irreps_in=irreps_in, irreps_out=irreps_out, biases=True)

        elif self.head_type == 'gated':

            irreps_scalars, irreps_gated = self.split_irreps(irreps_in)
            irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")

            # 1. Apply a linear layer to convert the number of input scalars to the number of required gating scalars
            # the number of input scalars is equal to sphere_channels
            # the output 'irreps_gates' are the gating scalars

            # --> gate with the l=0 components:
            # input_scalars_irreps = Irreps(f"{self.sphere_channels}x0e")
            # self.lin_scalars = e3nn_Linear(irreps_in=input_scalars_irreps, irreps_out=irreps_gates)

            # --> gate with x_edge:
            # input_scalars_irreps = Irreps(f"{3*self.sphere_channels}x0e")
            # self.lin_scalars_x_edge = e3nn_Linear(irreps_in=input_scalars_irreps, irreps_out=irreps_gates)
            # self.act_input_scalars = torch.nn.Sigmoid() 

            # --> gate with learnable parameters by outputting more random scalars:
            input_scalars_irreps = Irreps(f"{self.sphere_channels}x0e")
            combined_output_scalars = Irreps(f"{irreps_scalars.num_irreps + irreps_gated.num_irreps}x0e")
            self.lin_scalars_learnable = e3nn_Linear(irreps_in=input_scalars_irreps, irreps_out=combined_output_scalars, biases=True) # just a linear layer
            # self.lin_scalars_learnable_2 = nn.Linear(
            #     in_features=self.sphere_channels, 
            #     out_features=combined_output_scalars.num_irreps, 
            #     bias=True
            # )
            self.act_input_scalars = torch.nn.SiLU() # torch.nn.Sigmoid() # torch.nn.Tanh() # torch.nn.ReLU() # torch.nn.SiLU() # torch.nn.GELU()
            # this returns the irreps_gates

            # 2. Apply the gating to the other ls (need to pass in a stack of [l=0, l~=0])
            self.gate = Gate(irreps_scalars=Irreps(),
                                act_scalars=[],
                                irreps_gates=irreps_gates,
                                act_gates=[torch.sigmoid] * len(irreps_gates),
                                irreps_gated=irreps_gated
                            )
            # self.gate2 = Gate(irreps_scalars=Irreps(),
            #                     act_scalars=[],
            #                     irreps_gates=irreps_gates,
            #                     act_gates=[torch.sigmoid] * len(irreps_gates),
            #                     irreps_gated=irreps_gated
            #                 )
            # print("gate irreps out (simplified): ", self.gate.irreps_out.sort()[0].simplify() ) 
            # print("irreps out (simplified): ", irreps_out.sort()[0].simplify() ) 

            # now we have the [l=0s, gated l>0s] in a stack, and we just need to map them to the output irrep order:
            if reduce_node:
                self.lin_out_node = e3nn_Linear(irreps_in=irreps_scalars+self.gate.irreps_out, irreps_out=self.irreps_nodereduced, biases=True)
                self.lin_out_edge = e3nn_Linear(irreps_in=irreps_scalars+self.gate.irreps_out, irreps_out=self.irreps_out, biases=True) 
            else:
                self.lin_out = e3nn_Linear(irreps_in=irreps_scalars+self.gate.irreps_out, irreps_out=irreps_out, biases=True) 


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
        x_edge = emb["x_edge"]
        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)

        reverse_edge_map = batch.reverse_edge_map if "reverse_edge_map" in batch else None
        edge_mask = batch.edge_mask if "edge_mask" in batch else None

        if self.head_type == 'linear':
            node_embeddings = self.stack_irreps(node_embeddings)
            edge_embeddings = self.stack_irreps(edge_embeddings)
            node_output = self.map_node_to_rank_N(node_embeddings)
            edge_output = self.map_edge_to_rank_N(edge_embeddings)

        elif self.head_type == 'gated':
            node_embeddings = self.stack_irreps(node_embeddings)
            edge_embeddings = self.stack_irreps(edge_embeddings)
            node_output = self.process(node_embeddings, x_edge, edge_index, 'node')
            edge_output = self.process(edge_embeddings, x_edge, edge_index, 'edge')
            # node_output = self.process_doublegated(node_embeddings, x_edge, edge_index, 'node')
            # edge_output = self.process_doublegated(edge_embeddings, x_edge, edge_index, 'edge')
        
        else:
            print("Error! Mispelt head type")
        
        # augment the node irreps back to the full irrep list (containing the lower triangle of orbital interactions and odd self-interaction irreps)
        # using edge_output to infer the total size of the output embeddings
        if self.reduce_node:
            node_output = self.expand_reduced_node(node_output, edge_output)
        
        # need reflection on same device, could not access device from within constructor functions
        # self.edge_m_reflection = torch.tensor(self.edge_m_reflection, dtype=edge_output.dtype, device=edge_output.device)

        # # # Permute+reflect the irreps for the 'reverse' edges (the edge irreps are the same, but the order is different)
        # # NOTE: vectorize this later! 
        # if not (~edge_mask).any():              # if we are considering all the edges
        #     for i in range(len(edge_index[0])):
        #         source = edge_index[0][i]
        #         target = edge_index[1][i]
                
        #         # NOTE: source > target uniquely defines the direction of the edge!!!
        #         if torch.sum(node_output[source]) > torch.sum(node_output[target]): 
        #         # if i != reverse_edge_map[i]: # if this is a backward edge
        #             edge_output[i] = edge_output[reverse_edge_map[i], self.edge_permutation] * self.edge_m_reflection
        #             # edge_output[i] = edge_output[i, self.edge_permutation] * self.edge_m_reflection
        
        return node_output, edge_output

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

        for i, l1 in enumerate(self.ls_list):
            for j, l2 in enumerate(self.ls_list):

                # --> 1. Handle the permutation of the irreps:
                product_irreps = str(self.get_product_irreps(l1, l2))
                irrep_len = sum([2*l + 1 for l in Irreps(product_irreps).ls])

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

        # print("edge_permutation: ", edge_permutation)
                
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

    def process(self, x, x_edge, edge_index, node_or_edge):

        # 1. Extract the scalar components, which are the first # sphere_channels elements of this tensor
        x_scalars = x[:, :self.sphere_channels]
        x_nonscalars = x[:, self.sphere_channels:]

        # 2. Prepare some scalars for gating

        # x_for_gating = torch.zeros(
        #     (x.shape[0],) + x_edge.shape[1:],
        #     dtype=x.dtype,
        #     device=x.device,
        # )

        # # in case this is a node-block (x_edge is initially the edges, so it needs to be reduced from # edges to # nodes)
        # if x_edge.shape[0] != x.shape[0]:
        #     x_for_gating.index_add_(0, edge_index[1], x_edge)
        # else:
        #     x_for_gating = x_edge

        # gating_scalars = self.lin_scalars(x_scalars)                  # gate with the l=0 components
        # gating_scalars = self.lin_scalars_x_edge(x_for_gating)        # gate with x_edge

        # gate with learnable scalars: the first 'sphere_channels' scalars are the l=0, and others are used for gating
        all_scalars = self.lin_scalars_learnable(x_scalars) 
        transformed_l0_scalars = all_scalars[:, :self.sphere_channels]
        gating_scalars = all_scalars[:, self.sphere_channels:]

        # Symmetrize the final scalar values across edges, since they will be the same for both forward and backward edges
        # transformed_l0_scalars = torch.abs(transformed_l0_scalars) 
        # transformed_l0_scalars = self.symmetrize_scalars(transformed_l0_scalars)  

        # 3. Gate the l>0 irreps:
        x_gated = self.gate(torch.cat([gating_scalars, x_nonscalars], dim=1))
        x_gated = torch.cat([transformed_l0_scalars, x_gated], dim=1)   # use the transformed scalars as the output

        if not self.reduce_node:
            x_out = self.lin_out(x_gated)
        else:
            if node_or_edge == 'node':
                x_out = self.lin_out_node(x_gated)
            if node_or_edge == 'edge':
                x_out = self.lin_out_edge(x_gated)

        return x_out

    def process_doublegated(self, x, x_edge, edge_index, node_or_edge):

        # 1. Extract the scalar components, which are the first # sphere_channels elements of this tensor
        x_scalars = x[:, :self.sphere_channels]
        x_nonscalars = x[:, self.sphere_channels:]

        # 2. Prepare some scalars for gating
        # gate with learnable scalars: the first 'sphere_channels' scalars are the l=0, and others are used for gating
        all_scalars = self.lin_scalars_learnable(x_scalars) 

        # 3. Gate the l>0 irreps:

        # first gating pass:
        gating_scalars = all_scalars[:, self.sphere_channels:]
        x_gated = self.gate(torch.cat([gating_scalars, x_nonscalars], dim=1))

        # second gating pass:
        all_scalars_2 = self.lin_scalars_learnable_2(x_scalars)
        gating_scalars_2 = all_scalars_2[:, self.sphere_channels:]  # these are the scalars used for gating the second pass
        x_gated_2 = self.gate2(torch.cat([gating_scalars_2, x_gated], dim=1)) 

        # plug the l=0 components back into x_gated (currently they are zeros):
        transformed_l0_scalars = all_scalars[:, :self.sphere_channels]
        x_gated_out = torch.cat([transformed_l0_scalars, x_gated_2], dim=1)   # use the transformed scalars
        
        if not self.reduce_node:
            x_out = self.lin_out(x_gated_out)
        else:
            if node_or_edge == 'node':
                x_out = self.lin_out_node(x_gated_out)
            if node_or_edge == 'edge':
                x_out = self.lin_out_edge(x_gated_out)

        return x_out

    def expand_reduced_node(self, node_output, edge_output):
        """
        Expand irreps_nodereduced to irreps_out, by adding the previously-removed irreps back in:
        1. The odd irreps for the orbital self-interactions
        2. The 'lower triangle' of inter-orbital interactions on this node (eg the p-s to s-p)
        """
        assert self.orbital_basis is not None

        expanded_node_output = torch.zeros(
            (node_output.shape[0],) + edge_output.shape[1:],
            dtype=node_output.dtype,
            device=node_output.device,
        )

        output_irrep_p = 0                # pointer to track the irreps in irreps_out (from 0 to len(self.irreps_out)
        reduced_irrep_p = 0               # pointer to track the irreps in reduced_irreps

        for i, l1 in enumerate(self.ls_list):
            for j, l2 in enumerate(self.ls_list):
                
                # if it's a node, need to slot in only the even irrep components
                if i == j and l1 == l2 and self.reduce_node_intra:

                    even_irreps = self.get_product_irreps(l1, l2, 'even')
                    even_irreps_len = sum([2*l + 1 for l in even_irreps.ls])
                    odd_irreps = self.get_product_irreps(l1, l2, 'odd')
                    combined_irreps = Irreps(even_irreps + odd_irreps)    

                    # Extract even irreps:
                    local_p_output = 0  
                    local_p_reduced = 0  
                    for even_l in even_irreps.ls:
                        this_irrep_len = 2 * even_l + 1
                        expanded_node_output[:, output_irrep_p+local_p_output:output_irrep_p+local_p_output+this_irrep_len] = node_output[:, reduced_irrep_p+local_p_reduced:reduced_irrep_p+local_p_reduced+this_irrep_len]

                        # move start positions to the next even irrep
                        local_p_output += this_irrep_len
                        local_p_output += 2 * (even_l+1) + 1
                        local_p_reduced += this_irrep_len

                    # update output_irrep_pointer with combined_irreps size
                    output_irrep_p += sum([2*l + 1 for l in combined_irreps.ls])

                    # update reduced_irrep_pointer with even_irreps size
                    reduced_irrep_p += sum([2*l + 1 for l in even_irreps.ls])
                
                # if it's a forward edge, copy the next irreps directly from node_output
                if i < j or (i == j and l1 == l2 and not self.reduce_node_intra):      
                    these_irreps = self.get_product_irreps(l1, l2)
                    irreps_len = sum([2*l + 1 for l in these_irreps.ls])
                    expanded_node_output[:, output_irrep_p:output_irrep_p + irreps_len] = node_output[:, reduced_irrep_p:reduced_irrep_p + irreps_len]

                    output_irrep_p += irreps_len
                    reduced_irrep_p += irreps_len

                # if it's a backward edge, copy the corresponding forward edge (the location was saved in backward_irrep_track)
                if i > j:
                    these_irreps = self.get_product_irreps(l1, l2)
                    irreps_len = sum([2*l + 1 for l in these_irreps.ls])
                    forward_edge_bounds = self.backward_irrep_track[(i, j)] # contains the location of the forward edge

                    forward_edge_irreps = node_output[:, forward_edge_bounds[0]:forward_edge_bounds[1]]

                    # add the parity operator 
                    outer_parity = ((-1) ** (l1+l2)).item()
                    start_l = 0
                    for l in these_irreps.ls:
                        inner_parity = (-1) ** l
                        end_l = start_l + (2 * l + 1)                        
                        if inner_parity != outer_parity:
                            forward_edge_irreps[:, start_l:end_l] *= -1                        
                        start_l = end_l
                
                    expanded_node_output[:, output_irrep_p:output_irrep_p + irreps_len] = forward_edge_irreps
                    
                    output_irrep_p += sum([2*l + 1 for l in these_irreps.ls]) # only need to update the pointer along the output_irreps

        # print("node_output[0]: ", node_output[0])
        # print("expanded_node_output[0]: ", expanded_node_output[0])
        return expanded_node_output


@registry.register_model("esen_linear_fock_head")
class Seperable_Linear_Fock_Head(nn.Module):
        """
        Linear mapping from irreps of type Ex0e + Ex1e + Ex2e... (where E = sphere_channels) 
        to outputs like:

        layer 1: Ex0e + Ex1e + Ex2e ... -> Nsx0e (where Ns is the s-multiplicity of the output irreps) 
        layer 2: Ex0e + Ex1e + Ex2e ... -> Npx1e (where Np is the p-multiplicity of the output irreps)
        layer 3: Ex0e + Ex1e + Ex2e ... -> Ndx1e (where Nd is the d-multiplicity of the output irreps)
        ... 

        The output has the form Nsx0e+Npx1e+Ndx2e ... and then there is a final linear layer:
        --> e3nn_Linear(irreps_in="Nsx0e+Npx1e+Ndx2e", irreps_out=self.irreps_out, biases=False)
        """
        def __init__(self, sphere_channels, irreps_in, irreps_out, mappingReduced):
            super().__init__()
            self.irreps_in = irreps_in
            self.irreps_out = irreps_out
            self.mappingReduced = mappingReduced
            self.lmax = irreps_in.lmax
            self.mmax = irreps_in.lmax

            # groups the irreps of the same l in the output
            simplified_output_irreps = irreps_out.sort()[0].simplify()
            
            # Multiplicity of different ls in the output tensor (Ns, Np, Nd...)
            output_multiplicities = [int(str(ir).split('x')[0]) for ir in simplified_output_irreps]

            # Linear layers to handle the mappings for each l-component
            self.l_wise_linear_layers = nn.ModuleList()
            for l, mul in enumerate(output_multiplicities):
                l_output_irreps = Irreps(str(mul)+"x"+str(l)+"e")
                self.l_wise_linear_layers.append(e3nn_Linear(irreps_in, l_output_irreps, biases=True))

            # final linear layer to make output irrep order:
            self.convert_to_output_irreps = e3nn_Linear(irreps_in=simplified_output_irreps, irreps_out=self.irreps_out, biases=True)

        def forward(self, x):
            out = []
            for l_wise_linear_layer in self.l_wise_linear_layers:
                x_l = l_wise_linear_layer(x)
                out.append(x_l)
            out = torch.cat(out, dim=1)
            
            # convert from simplified to full output irreps        
            return self.convert_to_output_irreps(out)

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
        edge_mask = batch.edge_mask
        
        # Trim the embeddings to the chosen lmax:
        nodes = emb["node_embeddings"]#[:, :(self.lmax+1)**2, :]
        edges = emb["edge_embeddings"]#[:, :(self.lmax+1)**2, :]
        x_edge = emb["x_edge"]
        wigner = emb["wigner"]#[:, :(self.lmax+1)**2, :(self.lmax+1)**2]
        wigner_inv = emb["wigner_inv"]#[:, :(self.lmax+1)**2, :(self.lmax+1)**2]

        # Create the messages for the last convolution:
        x_source = nodes[edge_index[0][edge_mask]]
        x_target = nodes[edge_index[1][edge_mask]]
        if self.include_edges:
            # x_message = torch.cat((x_source, x_target, edges), dim=2) 
            x_message = torch.cat((x_source, x_target), dim=2) 
        else:
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
        new_embedding.index_add_(0, edge_index[1][edge_mask], x_message) # do this only for the first row
        energies = new_embedding.narrow(1, 0, 1)
        energies = self.linear(energies) 

        # -----------------

        # Edge convolution:
        # -----------------

        # edges = torch.bmm(emb["wigner"], edges)
        # edges = self.so2_conv_2(edges, x_edge)
        # edges = torch.bmm(wigner_inv, edges)
        # x.index_add_(0, edge_index[1][edge_mask], edges)  # aggregate the transformed edges onto the target nodes
        # x = self.linear(x)                                # apply the linear layer to the node embeddings
        # energies = x.narrow(1, 0, 1)                      # take the first channel as the energy

        energies = energies.squeeze(-1).squeeze(-1)  # remove the last two dimensions

        molecule_indices = torch.cat([
            torch.full((num_atoms,), i, dtype=torch.long, device=energies.device)
            for i, num_atoms in enumerate(batch.num_atoms_in_molecule)
        ])

        # atom-resolved energy -> molecule-resolved energy
        num_molecules = batch.num_atoms_in_molecule.size(0)
        energies_per_molecule = torch.zeros(num_molecules, device=energies.device)
        energies_per_molecule = energies_per_molecule.scatter_add(0, molecule_indices, energies)

        return {"energies": energies_per_molecule}

        # aggregate the edges onto every node:
        # if self.include_edges:
        #     agg_embedding = torch.zeros(
        #         (emb["node_embeddings"].shape[0],) + emb["edge_embeddings"].shape[1:],
        #         dtype=emb["node_embeddings"].dtype,
        #         device=emb["node_embeddings"].device,
        #     )

        #     agg_embedding.index_add_(0, edge_index[1], emb["node_embeddings"]) 
        # else:
        #     agg_embedding = emb["node_embeddings"]
        # energies = self.linear(agg_embedding.narrow(1, 0, 1))

        # final_node_output = self.conv(
        #                         emb["node_embeddings"],
        #                         emb["edge_embeddings"],
        #                         emb["x_edge"],
        #                         forward_edges,
        #                         edge_distance,
        #                         edge_index,
        #                         edge_mask,
        #                         reverse_edge_map,
        #                         emb["wigner"],
        #                         emb["wigner_inv"],
        #                         node_or_edge='node', 
        #                     )
        # energies = self.linear(final_node_output.narrow(1, 0, 1))


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
    
class Convolution_Force_Head(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        # self.output_node_block = eSEN_Block(
        #                                     backbone.sphere_channels,
        #                                     backbone.hidden_channels,
        #                                     backbone.lmax,
        #                                     backbone.mmax,
        #                                     backbone.mappingReduced,
        #                                     backbone.SO3_grid,
        #                                     backbone.edge_channels_list,
        #                                     backbone.cutoff,
        #                                     backbone.norm_type,
        #                                     backbone.act_type,
        #                                     backbone.mlp_type,
        #                                 )
        self.edgewise_forward = True
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, emb: dict[str, torch.Tensor], batch):

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)
        edge_distance = batch.edge_attr
        edge_mask = batch.edge_mask
        reverse_edge_map = batch.reverse_edge_map

        # edgewise forward (convolution + edgewise linear + aggregation):
        # -------------------------------------------------------
        if self.edgewise_forward:
            # final_edge_output = self.output_node_block(
            #         emb["node_embeddings"],
            #         emb["edge_embeddings"],
            #         emb["x_edge"],
            #         edge_distance,
            #         edge_index,
            #         edge_mask,
            #         reverse_edge_map,
            #         emb["wigner"],
            #         emb["wigner_inv"],
            #         node_or_edge='edge', 
            #     )
            final_edge_output = emb["edge_embeddings"]

            edgewise_forces = self.linear(final_edge_output.narrow(1, 0, 4))
            edgewise_forces = edgewise_forces.narrow(1, 1, 3)

            # aggregate force components onto nodes:
            aggregated_forces = torch.zeros(
                (emb["node_embeddings"].shape[0],) + edgewise_forces.shape[1:],
                dtype=edgewise_forces.dtype,
                device=edgewise_forces.device,
            )

            aggregated_forces.index_add_(0, edge_index[1][edge_mask], edgewise_forces)
            if (~edge_mask).any():                                              # if we are ignoring half the edges, need to now add the other half
                    aggregated_forces.index_add_(0, edge_index[0][edge_mask], -1*edgewise_forces)   


        # nodewise forward (convolution + aggregation):
        # -------------------------------------------------------
        else:

            aggregated_forces = self.output_node_block(
                    emb["node_embeddings"],
                    emb["edge_embeddings"],
                    emb["x_edge"],
                    edge_distance,
                    edge_index,
                    edge_mask,
                    reverse_edge_map,
                    emb["wigner"],
                    emb["wigner_inv"],
                    node_or_edge='node', 
                )

            aggregated_forces = self.linear(aggregated_forces.narrow(1, 0, 4))
            aggregated_forces = aggregated_forces.narrow(1, 1, 3)

        aggregated_forces = aggregated_forces.view(-1, 3).contiguous()

        # check that the forces are conserved:
        assert torch.allclose(torch.sum(aggregated_forces), torch.tensor(0.0), atol=1e-8), f"Force conservation check failed! Edge sum: {torch.sum(aggregated_forces)}"

        return {"forces": aggregated_forces}
