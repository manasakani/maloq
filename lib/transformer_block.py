import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
import copy
from mpi4py import MPI
import torch.distributed as dist
import numpy as np
import utils

# Note: we only use Gate Activation in this implementation
from activation import (
    SmoothLeakyReLU, 
    GateActivation,
)

from layer_norm import (
    get_normalization_layer
)

from SO2_operations import (
    SO2_Convolution, 
)

from SO3 import (
    SO3_Embedding, 
    SO3_LinearV2
)
from radial_function import RadialFunction

from torch import cuda
import os
import time
DEBUG = os.environ.get("DEBUG", False)

# TODO: remove this and make cleaner
local_stream = torch.cuda.Stream()
remote_stream = torch.cuda.Stream()


# Borrowed from EquiformerV2 (https://github.com/atomicarchitects/equiformer_v2.git)
class FeedForwardNetwork(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax (int):                 Max degree (l) 
        mmax (int):                 Max order (m) 

        activation (str):           Type of activation function
        
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels, 
        output_channels,
        lmax,
        mmax,
        activation='scaled_silu', 
    ):
        super(FeedForwardNetwork, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax = lmax
        self.mmax = mmax
        
        self.so3_linear_1 = SO3_LinearV2(self.sphere_channels, self.hidden_channels, lmax=self.lmax)
        self.so3_linear_2 = SO3_LinearV2(self.hidden_channels, self.output_channels, lmax=self.lmax)

        self.gating_linear = torch.nn.Linear(self.sphere_channels, self.lmax * self.hidden_channels)
        self.gate_act = GateActivation(self.lmax, self.lmax, self.hidden_channels)
        
    
    def forward(self, input_embedding):

        gating_scalars = None
        
        if self.gating_linear is not None:
            gating_scalars = self.gating_linear(input_embedding.embedding.narrow(1, 0, 1))

        input_embedding = self.so3_linear_1(input_embedding)
        
        input_embedding.embedding = self.gate_act(gating_scalars, input_embedding.embedding)
        
        input_embedding = self.so3_linear_2(input_embedding)

        return input_embedding       


# Borrowed from EquiformerV2 (https://github.com/atomicarchitects/equiformer_v2.git)
class SO2NodeUpdate(torch.nn.Module):
    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        num_heads, 
        attn_alpha_channels,
        attn_value_channels, 
        output_channels,
        lmax,
        mmax,
        SO3_rotation, 
        mappingReduced, 
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True, 
        use_m_share_rad=False,
        activation='scaled_silu', 
        use_attn_renorm=True,
    ):
        super(SO2NodeUpdate, self).__init__()
        
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax = lmax
        self.mmax = mmax
        
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        
        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        self.use_attn_renorm    = use_attn_renorm
                
        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
        extra_m0_output_channels = extra_m0_output_channels + self.lmax * self.hidden_channels # for gate activation
            
        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [3 * self.sphere_channels * (self.lmax + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(self.lmax + 1) ** 2]).long()
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

        self.so2_conv_1 = SO2_Convolution(
            3 * self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=(
                False if not self.use_m_share_rad 
                else True
            ),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad 
                else None
            ), 
            extra_m0_output_channels=extra_m0_output_channels # for attention weights and/or gate activation
        )

        if self.use_attn_renorm:
            self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
        else:
            self.alpha_norm = torch.nn.Identity()
        self.alpha_act = SmoothLeakyReLU()
        self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
        #torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        std = 1.0 / math.sqrt(self.attn_alpha_channels)
        torch.nn.init.uniform_(self.alpha_dot, -std, std)
        
        self.gate_act = GateActivation(
            lmax=self.lmax, 
            mmax=self.mmax, 
            num_channels=self.hidden_channels
        )
        
        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None, 
            extra_m0_output_channels=None
        )

        self.proj = SO3_LinearV2(self.num_heads * self.attn_value_channels, self.output_channels, lmax=self.lmax)


    def message_wise_operations(self, x_message, x_edge, start_edge, end_edge):

         # print("_________________________")
        # print("start of radial function")
        # print("_________________________")
        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_push("Radial function")


        # radial function (linear layers + layer normalization + SiLU)
        if self.use_m_share_rad:

            if DEBUG:
                cuda.current_stream().synchronize()
                cuda.nvtx.range_push("rad_func")

            x_edge_weight = self.rad_func(x_edge)

            if DEBUG:
                cuda.current_stream().synchronize()
                cuda.nvtx.range_pop()
    
            x_edge_weight = x_edge_weight.reshape(-1, (self.lmax + 1), 3 * self.sphere_channels) # 3x for the 3 concatenated features
            x_edge_weight = torch.index_select(x_edge_weight, dim=1, index=self.expand_index) # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # print("_________________________")
        # print("First rotation")
        # print("_________________________")

        # Rotate the irreps to align with the edge
        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_push("Rotate")
    
        x_message._rotate(self.SO3_rotation, self.lmax, self.mmax, start_edge, end_edge)

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("so2 conv 1")

        # print("_________________________")
        # print("SO2 Convolution 1")
        # print("_________________________")

        # First SO(2)-convolution
        x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # print("_________________________")
        # print("Gate Activation")
        # print("_________________________")

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("activation")

        # Activation (Gate activation)
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels) # for activation
        x_0_alpha  = x_0_extra.narrow(1, 0, x_alpha_num_channels) # for attention weights, shape [E, num_heads * attn_alpha_channels]
        x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)

        # print("_________________________")
        # print("SO2 Convolution 2")
        # print("_________________________")

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()           
            cuda.nvtx.range_push("so2 conv 2")

        # Second SO(2)-convolution
        x_message = self.so2_conv_2(x_message, x_edge)

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()

        return x_message.embedding, x_0_alpha

        
        
    def forward(
        self,
        x,
        partition,
        edge_distance,
        edge_index,
        global_edge_index,
        edge_fea,
        iteration
    ):

        # _______________________________________________________________________
        atomic_numbers = partition.global_atomic_numbers
        local_edge_idx = partition.local_edge_idx
        # _______________________________________________________________________

        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_push("Init source and target")

        start_message_creation = time.time()
        source_element = atomic_numbers[edge_index[0]]  
        target_element = atomic_numbers[edge_index[1]]  
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)
        x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)      # shape of [#edges, 3 * #channels]

        x_source = x.clone()
        x_target = x.clone()

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("Expand Edge 1")

        x_source._expand_edge(edge_index[0, :], partition.expand_edge_0)

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("Expand Edge 2")

        x_target._expand_edge(edge_index[1, :], partition.expand_edge_1)

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()
        end_message_creation = time.time()
        print(f"Time taken for message creation: {end_message_creation - start_message_creation}")

        # to form the message, concatenate the embeddings of the source node, target node, and the edge between them
        x_message_data = torch.cat((x_source.embedding, x_target.embedding, edge_fea.embedding), dim=2) 
        x_message = SO3_Embedding(
            0,
            x_target.lmax, 
            x_target.num_channels * 3, 
            device=x_target.device, 
            dtype=x_target.dtype
        )
        x_message.set_embedding(x_message_data)                                                # shape of [#edges, #channels, 3 * #channels]
        x_message.set_lmax_mmax(self.lmax, self.mmax)

        # Perform message-wise operations: Radial function, rotations, SO2-convolutions, gate activation, attention weights, and aggregation
        start_messagewise_ops = time.time()
        x_message_embedding, x_0_alpha = self.message_wise_operations(x_message, x_edge, 0, edge_fea.embedding.shape[0])
        end_messagewise_ops = time.time()
        print(f"Time taken for message-wise operations: {end_messagewise_ops - start_messagewise_ops}")

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_push("attention weights")

        x_message = SO3_Embedding(
            0,
            x_target.lmax, 
            x_target.num_channels * 3, 
            device=x_target.device, 
            dtype=x_target.dtype
        )
        x_message.set_embedding(x_message_embedding)
        x_message.set_lmax_mmax(self.lmax, self.mmax)

        # print("_________________________")
        # print("Attention weights")
        # print("_________________________")
        
        # Attention weights
        start_attention = time.time()
        x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels) # shape of [E, num_heads, attn_alpha_channels]
        x_0_alpha = self.alpha_norm(x_0_alpha)
        x_0_alpha = self.alpha_act(x_0_alpha)
        alpha = torch.einsum('bik, ik -> bi', x_0_alpha, self.alpha_dot)

        # Compute the softmax over the incoming edges
        offset_local_dst_indices = partition.expand_edge_0["local_indices"]
        alpha = torch_geometric.utils.softmax(alpha, offset_local_dst_indices)      # softmax over the incoming edges
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)                 # shape of [E, 1, num_heads, 1]

        # Attention weights * non-linear messages (weight each message by the corresponding attention weight)
        attn = x_message.embedding                                                                      # shape of [E, (lmax+1)^2, # hidden channels]
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)     # shape of [E, #channels, num_heads, attn_value_channels]
        attn = attn * alpha
        attn = attn.reshape(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
        x_message.embedding = attn
        end_attention = time.time()
        print(f"Time taken for attention mechanism: {end_attention - start_attention}")

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("Rotate Inv")

        # print("_________________________")
        # print("Second Rotation")
        # print("_________________________")

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced, 0, x_message.embedding.shape[0])

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("Reduce edge")

        # print("_________________________")
        # print("Aggregation")
        # print("_________________________")

        # Aggregate incoming neighboring messages for each target node
        x_message._reduce_edge(edge_index[0], local_edge_idx, len(x.embedding), partition.reduce_edge)

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("proj")

        # Project
        node_embedding = self.proj(x_message)

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()

        return node_embedding
    
# Adapted from EquiformerV2 (https://github.com/atomicarchitects/equiformer_v2.git)
class NodeBlockV2(torch.nn.Module):
    def __init__(
        self,
        sphere_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels, 
        attn_value_channels,
        ffn_hidden_channels,
        output_channels, 
        lmax,
        mmax,
        SO3_rotation,
        mappingReduced,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        use_m_share_rad=False,
        attn_activation='silu',
        use_attn_renorm=True,
        ffn_activation='silu',
        norm_type='rms_norm_sh',
    ):
        super(NodeBlockV2, self).__init__()

        assert sphere_channels == output_channels, "NodeBlockV2 only supports sphere_channels == output_channels"

        self.norm_1 = get_normalization_layer(norm_type, lmax=lmax, num_channels=sphere_channels)
        self.norm_2 = get_normalization_layer(norm_type, lmax=lmax, num_channels=sphere_channels)
        
        self.ga = SO2NodeUpdate(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads, 
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels, 
            output_channels=output_channels,
            lmax=lmax,
            mmax=mmax,
            SO3_rotation=SO3_rotation, 
            mappingReduced=mappingReduced, 
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding, 
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation, 
            use_attn_renorm=use_attn_renorm,
        )

        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels, 
            output_channels=output_channels,
            lmax=lmax,
            mmax=mmax,
            activation=ffn_activation,
        )

    
    def forward(
        self,
        x,                              # node embedding
        partition,
        edge_distance,                  # edge distance embedding (initial edge features)
        edge_index,
        global_edge_index,
        edge_fea,                       # edge embedding
        iteration
    ):

        output_embedding = x
        x_res = output_embedding.embedding

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_push("norm_1")

        # Normalize the input embedding (done independantly for each embedding)
        output_embedding.embedding = self.norm_1(output_embedding.embedding)

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()

        # Perform the SO2NodeUpdate
        output_embedding = self.ga(output_embedding, 
            partition,
            edge_distance,
            edge_index, global_edge_index, edge_fea, iteration)
    
        # Add the residual connection and update the output embedding
        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_push("norm_2")

        output_embedding.embedding = output_embedding.embedding + x_res
        x_res = output_embedding.embedding

        # Normalize the output embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("ffn")

        # Pass through the feedforward network
        output_embedding = self.ffn(output_embedding)

        # Add the residual connection
        output_embedding.embedding = output_embedding.embedding + x_res

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()

        return output_embedding
    

# Adapted from EquiformerV2 (https://github.com/atomicarchitects/equiformer_v2.git)
class SO2EdgeUpdate(torch.nn.Module):

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        num_heads, 
        attn_alpha_channels,
        attn_value_channels, 
        output_channels,
        lmax,
        mmax,
        SO3_rotation, 
        mappingReduced, 
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True, 
        use_m_share_rad=True,
        activation='scaled_silu', 
        use_attn_renorm=True,
    ):
        super(SO2EdgeUpdate, self).__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax = lmax
        self.mmax = mmax
        
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        
        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None        
        
        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
        extra_m0_output_channels = extra_m0_output_channels + self.lmax * self.hidden_channels
            
        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [3 * self.sphere_channels * (self.lmax + 1)]
            self.rad_func = RadialFunction(self.edge_channels_list)

            expand_index = torch.zeros([(self.lmax + 1) ** 2]).long()
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

        self.so2_conv_1 = SO2_Convolution(
            3 * self.sphere_channels,
            self.hidden_channels,
            self.lmax,
            self.mmax,
            self.mappingReduced,
            internal_weights=(
                False if not self.use_m_share_rad 
                else True
            ),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad 
                else None
            ), 
            extra_m0_output_channels=extra_m0_output_channels # for attention weights and/or gate activation
        )

        self.gate_act = GateActivation(
            lmax=self.lmax, 
            mmax=self.mmax, 
            num_channels=self.hidden_channels
        )

        self.proj = SO3_LinearV2(self.hidden_channels, self.output_channels, lmax=self.lmax)


    def message_wise_operations(self, x_message, x_edge, start_edge, end_edge):    
        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_push("Radial function")

        # radial function (linear layers + layer normalization + SiLU)
        if self.use_m_share_rad:

            if DEBUG:
                cuda.current_stream().synchronize()
                cuda.nvtx.range_push("rad_func")

            x_edge_weight = self.rad_func(x_edge)

            if DEBUG:
                cuda.current_stream().synchronize()
                cuda.nvtx.range_pop()

            x_edge_weight = x_edge_weight.reshape(-1, (self.lmax + 1), 3 * self.sphere_channels)
            x_edge_weight = torch.index_select(x_edge_weight, dim=1, index=self.expand_index) # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_push("Rotate")

        # Rotate the irreps to align with the edge
        x_message._rotate(self.SO3_rotation, self.lmax, self.mmax, start_edge, end_edge)

        if DEBUG:
            cuda.nvtx.range_pop()
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("so2 conv 1")

        # First SO(2)-convolution
        x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("activation")

        # Activation (Gate activation)
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        x_0_gating = x_0_extra.narrow(1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels) # for activation
        x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("Rotate Inv")

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced, start_edge, end_edge)

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("proj")

        # Project
        edge_embedding = self.proj(x_message)

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()

        return edge_embedding

    def forward(
        self,
        x,
        partition,
        edge_distance,
        edge_index,
        global_edge_index,
        edge_fea
    ):

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_push("Init source and target")

        atomic_numbers = partition.global_atomic_numbers
         
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance  

        x_source = x.clone()
        x_target = x.clone()

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("Expand Edge 1")

        x_source._expand_edge(edge_index[0, :], partition.expand_edge_0)

        if DEBUG:
            cuda.synchronize()
            cuda.nvtx.range_pop()
            cuda.nvtx.range_push("Expand Edge 2")

        x_target._expand_edge(edge_index[1, :], partition.expand_edge_1)

        if DEBUG:
            cuda.current_stream().synchronize()
            cuda.nvtx.range_pop()

        # create the message by concatenating the embeddings of the source node, target node, and the edge between them
        x_message_data = torch.cat((x_source.embedding, x_target.embedding, edge_fea.embedding), dim=2) 
        x_message = SO3_Embedding(
            0,
            x_target.lmax, 
            x_target.num_channels * 3, 
            device=x_target.device, 
            dtype=x_target.dtype
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax, self.mmax)

        # Perform message-wise operations: Radial function, rotations, SO2-convolutions, gate activation, attention weights, and aggregation
        edge_embedding = self.message_wise_operations(x_message, x_edge, 0, edge_fea.embedding.shape[0])

        return edge_embedding


# Adapted from EquiformerV2 (https://github.com/atomicarchitects/equiformer_v2.git)
class EdgeBlockV2(torch.nn.Module):
    def __init__(
        self,
        sphere_channels,
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels, 
        attn_value_channels,
        ffn_hidden_channels,
        output_channels, 
        lmax,
        mmax,
        SO3_rotation,
        mappingReduced,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        use_m_share_rad=False,
        attn_activation='silu',
        use_attn_renorm=True,
        ffn_activation='silu',        
        norm_type='rms_norm_sh',
    ):
        super(EdgeBlockV2, self).__init__()

        assert sphere_channels == output_channels, "SO2EdgeUpdate only supports sphere_channels == output_channels"

        self.norm_1 = get_normalization_layer(norm_type, lmax=lmax, num_channels=sphere_channels)
        self.norm_2 = get_normalization_layer(norm_type, lmax=lmax, num_channels=sphere_channels)

        self.ga = SO2EdgeUpdate(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads, 
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels, 
            output_channels=output_channels,
            lmax=lmax,
            mmax=mmax,
            SO3_rotation=SO3_rotation, 
            mappingReduced=mappingReduced, 
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding, 
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation, 
            use_attn_renorm=use_attn_renorm,
        )
        
        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels, 
            output_channels=output_channels,
            lmax=lmax,
            mmax=mmax,
            activation=ffn_activation,
        )

    
    def forward(
        self,
        x,              # SO3_Embedding
        partition,
        edge_distance,
        edge_index,
        global_edge_index,
        edge_fea,
    ):

        output_embedding = edge_fea
        x_res = output_embedding.embedding

        # Normalize the input embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)

        # Perform the SO2EdgeUpdate
        output_embedding = self.ga(x,  # use the node embedding from the previous block 
            partition,
            edge_distance,
            edge_index, global_edge_index, output_embedding) # put the output_embedding in place of edge_embedding 
        
        # Add the residual connection and update the output embedding
        output_embedding.embedding = output_embedding.embedding + x_res
        x_res = output_embedding.embedding

        # Normalize the output embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)

        # Pass through the feedforward network
        output_embedding = self.ffn(output_embedding)

        # Add the residual connection
        output_embedding.embedding = output_embedding.embedding + x_res

        return output_embedding