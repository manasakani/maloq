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
from e3nn.o3 import Irreps 
from e3nn.o3 import Linear as e3nn_Linear
from e3nn.nn import Gate
from torch.nn import Linear

# Fix this later!:
sys.path.append('/home/manasakani/fairchem/src/')
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin, HeadInterface

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
from .nn.so3_layers import SO3_Linear

@registry.register_model("esen_backbone")
class eSEN_Backbone(nn.Module, GraphModelMixin):
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
        gaussian_width = 1.0
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.gaussian_width = gaussian_width

        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.regress_stress = regress_stress
        self.mlp_type = mlp_type

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
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
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
            )
            self.node_blocks.append(node_block)
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
            )
            self.edge_blocks.append(edge_block)

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.sphere_channels,
        )

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
            # "edge_index": torch.tensor(batch.edge_index, dtype=torch.long).squeeze(0).reshape(2, -1),
            "edge_index": batch.edge_index.squeeze(0).reshape(2, -1),
            "edge_dist": batch.edge_attr,
            "nedges": len(batch.edge_index[0]),
            "natoms": len(batch.pos),
            "atomic_numbers": batch.atomic_numbers
        }

        # The input edges are in xyz coordinates, we need to rotate them to the yzx coordinates expected by e3nn
        # edge_distance_vec = data_dict["edge_dist"][:, 0:3]  # assuming the edge distances are already in the form of yzx
        edge_distance_vec = data_dict["edge_dist"][:, [2, 3, 1]] 
        edge_distance = data_dict["edge_dist"][:, 0] 

        # # From original eSEN forward pass:
        # edge_distance_vec = (
        #     data_dict["pos"][data_dict["edge_index"][0]]
        #     - data_dict["pos"][data_dict["edge_index"][1]]
        # )
        # # pylint: disable=E1102
        # edge_distance_vec = edge_distance_vec[:, [1, 2, 0]] # rotate to yzx coordinates so the correct rotation is found
        # edge_distance = torch.linalg.norm(edge_distance_vec, dim=-1, keepdim=False)

        graph_dict = {
            "edge_index": data_dict["edge_index"],
            "edge_distance": edge_distance,
            "edge_distance_vec": edge_distance_vec,
        }

        edge_rot_mat, wigner, wigner_inv = self.get_rotmat_and_wigner(
            graph_dict["edge_distance_vec"]
        )

        # check rotation matrix:
        # rotated_edges_to_z_axis = torch.bmm(edge_rot_mat, graph_dict["edge_distance_vec"].unsqueeze(-1)).squeeze(-1)

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

        # x_message_edge: [data_dict["nedges"] = #edges, self.sph_feature_size = (l_max+1)**2, self.sphere_channels = E]
        x_message_edge = torch.zeros(
            data_dict["nedges"],
            self.sph_feature_size,
            self.num_distance_basis, #self.sphere_channels,
            device=data_dict["pos"].device,
            dtype=data_dict["pos"].dtype,
        )
        # set l = 0 components to the distance expansion
        x_message_edge[:, 0, :] = self.distance_expansion(graph_dict["edge_distance"])

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

        x_edge = torch.cat(
            (edge_distance_embedding, source_embedding, target_embedding), dim=1
        )

        # do edge degree embeddings for both nodes and edges:
        x_message_node = self.edge_degree_embedding(
            x_message_node,
            x_edge,
            graph_dict["edge_distance"],
            graph_dict["edge_index"],
            wigner_inv,
            node_or_edge='node'
        )

        x_message_edge = self.edge_degree_embedding(
            x_message_node,
            x_edge,
            graph_dict["edge_distance"],
            graph_dict["edge_index"],
            wigner_inv,
            node_or_edge='edge'
        )
        
        # x_message_node shape:    # nodes, lmax, E
        # x_message_edge shape:    # edges, lmax, E

        ###############################################################
        # Update spherical node embeddings
        ###############################################################
        for i in range(self.num_layers):
            x_message_node = self.node_blocks[i](
                x_message_node,
                x_message_edge,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner,
                wigner_inv,
                node_or_edge='node',
            )
            x_message_edge = self.edge_blocks[i](
                x_message_node,
                x_message_edge,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner,
                wigner_inv,
                node_or_edge='edge',
            )
        
        # Final layer norm
        x_message_node = self.norm(x_message_node)
        x_message_edge = self.norm(x_message_edge)

        # Print to output files:

        # if output_dir:
        #     file_path = os.path.join(output_dir, f'molecule_{batch_index}.txt')
        #     with open(file_path, 'w') as f:
        #         # Write the shape and elements of x_message_node
        #         f.write("x_message_node\n")
        #         f.write(f"{x_message_node.shape}\n")
        #         f.write(' '.join(map(str, x_message_node.flatten().tolist())) + "\n")
                
        #         # Write the shape and elements of x_message_edge
        #         f.write("x_message_edge\n")
        #         f.write(f"{x_message_edge.shape}\n")
        #         f.write(' '.join(map(str, x_message_edge.flatten().tolist())) + "\n")

        # Prepare rank-N outputs:
        # node_rank0 = self.convert_to_output_irreps(x_message_node, x_edge, self.sphere_channels, self.lmax, rank='0')
        # node_rank1 = self.convert_to_output_irreps(x_message_node, x_edge, self.sphere_channels, self.lmax, rank='1')
        # node_rankN = self.convert_to_output_irreps(x_message_node, x_edge, self.sphere_channels, self.lmax, rank='N', edge_index=graph_dict["edge_index"], node_or_edge='node') 
        # edge_rankN = self.convert_to_output_irreps(x_message_edge, x_edge, self.sphere_channels, self.lmax, rank='N', edge_index=graph_dict["edge_index"], node_or_edge='edge')

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
    def __init__(self, irreps_in, irreps_out, lmax, sphere_channels, head_type='gated'):
        super().__init__()

        self.head_type = head_type
        self.sphere_channels = sphere_channels
        self.lmax = lmax

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
            self.lin_scalars_learnable = e3nn_Linear(irreps_in=input_scalars_irreps, irreps_out=combined_output_scalars)

            # this returns the irreps_gates

            # self.gate = Gate(irreps_scalars=irreps_scalars,
            #                     act_scalars=[torch.tanh] * len(irreps_scalars),
            #                     irreps_gates=irreps_gates,
            #                     act_gates=[torch.tanh] * len(irreps_gates),
            #                     irreps_gated=irreps_gated
            #                 )

            # 2. Apply the gating to the other ls (need to pass in a stack of [l=0, l~=0])
            self.gate = Gate(irreps_scalars=Irreps(),
                                act_scalars=[],
                                irreps_gates=irreps_gates,
                                act_gates=[torch.sigmoid] * len(irreps_gates),
                                irreps_gated=irreps_gated
                            )
            # print("gate irreps out (simplified): ", self.gate.irreps_out.sort()[0].simplify() ) 
            # print("irreps out (simplified): ", irreps_out.sort()[0].simplify() ) 

            # now we have the [l=0s, gated l>0s] in a stack, and we just need to map them to the output irrep order:
            self.lin_out = e3nn_Linear(irreps_in=irreps_scalars+self.gate.irreps_out, irreps_out=irreps_out, biases=False) 

    
    def split_irreps(self, irreps):
        scalars = []
        gated = []
        for mul, ir in irreps:
            if ir.l == 0:
                scalars.append((mul, ir))
            else:
                gated.append((mul, ir))
        return Irreps(scalars), Irreps(gated)

    def forward(self, emb, batch):

        node_embeddings = emb["node_embeddings"]
        edge_embeddings = emb["edge_embeddings"]
        x_edge = emb["x_edge"]
        edge_index = torch.tensor(batch.edge_index, dtype=torch.long).squeeze(0).reshape(2, -1)

        if self.head_type == 'linear':
            node_embeddings = self.stack_irreps(node_embeddings)
            edge_embeddings = self.stack_irreps(edge_embeddings)
            node_output = self.map_node_to_rank_N(node_embeddings)
            edge_output = self.map_edge_to_rank_N(edge_embeddings)

        elif self.head_type == 'gated':
            node_embeddings = self.stack_irreps(node_embeddings)
            edge_embeddings = self.stack_irreps(edge_embeddings)
            node_output = self.process(node_embeddings, x_edge, edge_index)
            edge_output = self.process(edge_embeddings, x_edge, edge_index)
        
        else:
            print("Error! Mispelt head type")

        return node_output, edge_output

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

    def process(self, x, x_edge, edge_index):

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

        # 3. Gate the l>0 irreps:
        x_gated = self.gate(torch.cat([gating_scalars, x_nonscalars], dim=1))

        # plug the l=0 components back into x_gated (currently they are zeros):
        # x_gated = torch.cat([x_scalars, x_gated], dim=1)              # original scalars get plugged back in
        x_gated = torch.cat([transformed_l0_scalars, x_gated], dim=1)   # use the transformed scalars
        x_out = self.lin_out(x_gated)

        return x_out

# @registry.register_model("esen_nonlinear_fock_head")
# class Nonlinear_Fock_Head(nn.Module):
#         """
#         Nonlinear mapping from irreps of type Ex0e + Ex1e + Ex2e... (where E = sphere_channels) 
#         to rank_N output of arbitrary stack irreps_out?
#         """
#         def __init__(self, sphere_channels, irreps_in, irreps_out, mappingReduced):
#             super().__init__()
#             self.irreps_in = irreps_in
#             self.irreps_out = irreps_out
#             self.mappingReduced = mappingReduced
#             self.lmax = irreps_in.lmax
#             self.mmax = irreps_in.lmax

#             # group the ls for now, and they can be ungrouped later with an e3nn linear layer.
#             simplified_output_irreps = irreps_out.sort()[0].simplify()          
            
#             # get the multiplicities of different ls in the output tensor
#             output_multiplicities = [int(str(ir).split('x')[0]) for ir in simplified_output_irreps]
#             print("multiplicities in output Irreps: ", output_multiplicities)

#             # Linear layers to handle the mappings for each m-component
#             # mul is the l-multiplicity! need to change to the m multiplicity
#             self.m_wise_linear_layers = nn.ModuleList()
#             for m, mul in enumerate(output_multiplicities):
#                 if m == 0: 
#                     self.m_wise_linear_layers.append(torch.nn.Linear(self.mappingReduced.m_size[m]*sphere_channels, mul))
#                     # print("added matrix of shape ", torch.nn.Linear(self.mappingReduced.m_size[i]*sphere_channels, mul).weight.shape)
#                 else:
#                     self.m_wise_linear_layers.append(
#                     SO2_m_Conv_output(
#                         m,
#                         sphere_channels,
#                         mul,
#                         self.lmax,
#                         self.mmax,
#                     )
#                 )

#             # ----------------------
#             # ADD NONLINEARITY HERE?
#             # ----------------------

#             # -------------------------------------
#             # Permute output irreps back to l-major
#             # -------------------------------------

#             # final linear layer to make output irrep order:
#             # linear_to_output_irreps = e3nn_Linear(irreps_in=[current irreps], irreps_out=self.irreps_out, biases=True)


#         def forward(self, x): 

#             # move _to_m:
#             x = torch.einsum("nac,ba->nbc", x, self.mappingReduced.to_m) # now the first ones are m=0, the second are m=-1, etc
#             feature_dim = x.shape[0]
#             print("x shape ", x.shape)

#             print("size of ms: ", self.mappingReduced.m_size)

#             out = []

#             # Do m = 0 part:
#             x_0 = x.narrow(1, 0, self.mappingReduced.m_size[0])
#             print("x_0 size: ", x_0.shape)
#             x_0 = x_0.reshape(feature_dim, -1)
#             print("x_0 size: ", x_0.shape)
#             x_0 = self.m_wise_linear_layers[0](x_0)
#             print("x_0 size: ", x_0.shape)
#             x_0 = x_0.unsqueeze(1)
#             out.append(x_0)

#             # Do nonzero-m part:
#             offset = self.mappingReduced.m_size[0]
#             for m in range(1, self.mmax + 1):
#                 # Get the m order coefficients
#                 x_m = x.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
#                 print("x_m size: ", x_m.shape)
#                 x_m = x_m.reshape(feature_dim, 2, -1)
#                 print("x_m size: ", x_m.shape)

#                 # Perform SO(2) convolution
#                 x_m = self.m_wise_linear_layers[m](x_m)
#                 print("x_m size: ", x_m.shape)
#                 print("----------")
#                 # x_m = x_m.view(num_edges, -1, self.m_output_channels)
#                 out.append(x_m)
#                 offset = offset + 2 * self.mappingReduced.m_size[m]

#             print([o.shape for o in out])
#             # out = torch.cat(out, dim=1)
#             out = torch.cat([o for o in out], dim=1)
#             print("shape of out: ", out.shape)

#             # now the tensors have size [num_nodes, +/-m, num_ms]
#             # now the first ones are m=0, the second are m=-1, the third are m = -1

#             # apply non-linearity

#             # rearrange the output back to l-major

#             print("exiting")
#             exit()

#             # re-arrange the output back into unsorted output irreps with an e3nn linear layer


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


# class SO2_m_Conv_output(torch.nn.Module):
#     """
#     SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

#     Args:
#         m (int):                    Order of the spherical harmonic coefficients
#         sphere_channels (int):      Number of spherical channels
#         m_output_channels (int):    Number of output channels used during the SO(2) conv
#         lmax (int):                 degrees (l)
#         mmax (int):                 orders (m)
#     """

#     def __init__(
#         self,
#         m: int,
#         sphere_channels: int,
#         m_multiplicity: int,
#         lmax: int,
#         mmax: int,
#     ) -> None:
#         super().__init__()

#         self.m = m
#         self.sphere_channels = sphere_channels
#         # self.m_output_channels = m_output_channels
#         self.lmax = lmax
#         self.mmax = mmax

#         assert self.mmax >= m
#         num_coefficents = self.lmax - m + 1
#         num_channels = num_coefficents * self.sphere_channels

#         self.out_channels_half = m_multiplicity
#         # self.m_output_channels * (
#         #     num_channels // self.sphere_channels
#         # )
#         self.fc = Linear(
#             num_channels,
#             2 * self.out_channels_half,
#             bias=False,
#         )
#         self.fc.weight.data.mul_(1 / math.sqrt(2))

#     def forward(self, x_m):
#         x_m = self.fc(x_m)
#         x_r = x_m.narrow(2, 0, self.out_channels_half)
#         x_i = x_m.narrow(2, self.out_channels_half, self.out_channels_half)
#         x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)  # x_r[:, 0] - x_i[:, 1]
#         x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)  # x_r[:, 1] + x_i[:, 0]
#         return torch.cat((x_m_r, x_m_i), dim=1)

@registry.register_model("esen_mlp_efs_head")
class MLP_EFS_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        backbone.energy_block = None
        backbone.force_block = None
        self.regress_stress = backbone.regress_stress
        self.regress_forces = backbone.regress_forces

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

        backbone.direct_forces = False

    @conditional_grad(torch.enable_grad())
    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_key = "energy"
        forces_key = "forces"
        stress_key = "stress"

        outputs = {}

        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        energy = torch.zeros(
            len(data["natoms"]), device=data["pos"].device, dtype=node_energy.dtype
        )
        energy.index_add_(0, data["batch"], node_energy.view(-1))
        outputs[energy_key] = energy

        if self.regress_stress:
            grads = torch.autograd.grad(
                [energy.sum()],
                [data["pos"], emb["displacement"]],
                create_graph=self.training,
            )
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data["cell"]).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)
            outputs[forces_key] = forces
            outputs[stress_key] = stress.view(-1, 9)
            data["cell"] = emb["orig_cell"]
        elif self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy.sum(), data["pos"], create_graph=self.training
                )[0]
            )
            outputs[forces_key] = forces
        return outputs


@registry.register_model("esen_mlp_energy_head")
class MLP_Energy_Head(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce

        self.sphere_channels = backbone.sphere_channels
        self.hidden_channels = backbone.hidden_channels
        self.energy_block = nn.Sequential(
            nn.Linear(self.sphere_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, 1, bias=True),
        )

    def forward(self, data_dict, emb: dict[str, torch.Tensor]):
        node_energy = self.energy_block(
            emb["node_embedding"].narrow(1, 0, 1).squeeze()
        ).view(-1, 1, 1)

        energy = torch.zeros(
            len(data_dict["natoms"]),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy.index_add_(0, data_dict["batch"], node_energy.view(-1))
        if self.reduce == "sum":
            return {"energy": energy}
        elif self.reduce == "mean":
            return {"energy": energy / data_dict["natoms"]}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )


@registry.register_model("esen_linear_force_head")
class Linear_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        # self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)
        self.linear = SO3_Linear(2*backbone.sphere_channels, 1, lmax=1)

    def forward(self, emb: dict[str, torch.Tensor], batch):

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)
        
        aggregated_emb = torch.zeros_like(emb["node_embeddings"])
        aggregated_emb.index_add_(0, edge_index[1], emb["edge_embeddings"])
        node_plus_edges = torch.cat((emb["node_embeddings"], aggregated_emb), 2)            # concatenate the node with its aggregated edges:
        forces = self.linear(node_plus_edges.narrow(1, 0, 4))

        # forces = self.linear(emb["node_embeddings"].narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        return {"forces": forces}
    
class Convolution_Force_Head(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()
        
        self.output_node_block = eSEN_Block(
                                            backbone.sphere_channels,
                                            backbone.hidden_channels,
                                            backbone.lmax,
                                            backbone.mmax,
                                            backbone.mappingReduced,
                                            backbone.SO3_grid,
                                            backbone.edge_channels_list,
                                            backbone.cutoff,
                                            backbone.norm_type,
                                            backbone.act_type,
                                            backbone.mlp_type,
                                        )
        self.linear = SO3_Linear(backbone.sphere_channels, 1, lmax=1)

    def forward(self, emb: dict[str, torch.Tensor], batch):

        edge_index = batch.edge_index.squeeze(0).reshape(2, -1)
        edge_distance = batch.edge_attr

        aggregated_node_output = self.output_node_block(
                emb["node_embeddings"],
                emb["edge_embeddings"],
                emb["x_edge"],
                edge_distance,
                edge_index,
                emb["wigner"],
                emb["wigner_inv"],
                node_or_edge='node',
            )
        
        # plt.imshow(aggregated_node_output[1].cpu().detach().numpy(), cmap='RdBu', vmin=-1.0, vmax=1.0)
        # plt.savefig("internal_emb.png", dpi=300, bbox_inches='tight')
        # plt.close()
        # exit()

        forces = self.linear(aggregated_node_output.narrow(1, 0, 4))
        forces = forces.narrow(1, 1, 3)
        forces = forces.view(-1, 3).contiguous()
        return {"forces": forces}



@registry.register_model("gated_force_head")
class Gated_Force_Head(nn.Module):

    def __init__(self, backbone, irreps_in):
        super().__init__()

        self.sphere_channels = 2*backbone.sphere_channels
        self.lmax = backbone.lmax
        irreps_out = '1x1e'

        irreps_scalars, irreps_gated = self.split_irreps(irreps_in)
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")
        print("num ofirreps_gates: ", irreps_gates.num_irreps)

        # 1. Apply a linear layer to convert the number of input scalars to the number of required gating scalars
        # the number of input scalars is equal to sphere_channels
        # the output 'irreps_gates' are the gating scalars
        # --> gate with learnable parameters by outputting more random scalars:
        input_scalars_irreps = Irreps(f"{self.sphere_channels}x0e")
        combined_output_scalars = Irreps(f"{irreps_scalars.num_irreps + irreps_gated.num_irreps}x0e")
        self.lin_scalars_learnable = e3nn_Linear(irreps_in=input_scalars_irreps, irreps_out=combined_output_scalars)
        # this returns the irreps_gates

        # 2. Apply the gating to the other ls (need to pass in a stack of [l=0, l~=0])
        self.gate = Gate(irreps_scalars=Irreps(),
                            act_scalars=[],
                            irreps_gates=irreps_gates,
                            act_gates=[torch.sigmoid] * len(irreps_gates),
                            irreps_gated=irreps_gated
                        )

        # 3. Now we have the [l=0s, gated l>0s] in a stack, and we just need to map them to the output irrep order:
        self.lin_out = e3nn_Linear(irreps_in=irreps_scalars+self.gate.irreps_out, irreps_out=irreps_out, biases=False) 

    
    def split_irreps(self, irreps):
        scalars = []
        gated = []
        for mul, ir in irreps:
            if ir.l == 0:
                scalars.append((mul, ir))
            else:
                gated.append((mul, ir))
        return Irreps(scalars), Irreps(gated)

    def forward(self, emb, batch):

        x_edge = emb["x_edge"]
        edge_index = torch.tensor(batch.edge_index, dtype=torch.long).squeeze(0).reshape(2, -1)

        aggregated_emb = torch.zeros_like(emb["node_embeddings"])
        aggregated_emb.index_add_(0, edge_index[1], emb["edge_embeddings"])
        node_plus_edges = torch.cat((emb["node_embeddings"], aggregated_emb), 2)       

        node_embeddings = self.stack_irreps(node_plus_edges)
        node_output = self.process(node_embeddings, x_edge, edge_index)

        return node_output, edge_output

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

    def process(self, x, x_edge, edge_index):

        # 1. Extract the scalar components, which are the first # sphere_channels elements of this tensor
        x_scalars = x[:, :self.sphere_channels]
        x_nonscalars = x[:, self.sphere_channels:]

        # 2. Prepare some scalars for gating

        # gate with learnable scalars: the first 'sphere_channels' scalars are the l=0, and others are used for gating
        all_scalars = self.lin_scalars_learnable(x_scalars) 
        transformed_l0_scalars = all_scalars[:, :self.sphere_channels]
        gating_scalars = all_scalars[:, self.sphere_channels:]

        print(transformed_l0_scalars.shape)
        print(gating_scalars.shape)
        print(x_nonscalars.shape)
        exit()

        # 3. Gate the l>0 irreps:
        x_gated = self.gate(torch.cat([gating_scalars, x_nonscalars], dim=1))

        # plug the l=0 components back into x_gated (currently they are zeros):
        # x_gated = torch.cat([x_scalars, x_gated], dim=1)              # original scalars get plugged back in
        x_gated = torch.cat([transformed_l0_scalars, x_gated], dim=1)   # use the transformed scalars
        x_out = self.lin_out(x_gated)

        return x_out