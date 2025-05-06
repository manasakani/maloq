# from e3nn.o3 import Irrep, Irreps, wigner_3j, matrix_to_angles, Linear, FullyConnectedTensorProduct, TensorProduct, SphericalHarmonics
# from e3nn.nn import Extract
from e3nn.o3 import Irreps, wigner_3j
import torch
import numpy as np


def irreps_from_l1l2(l1, l2, mul, no_parity=True):
    r'''
    non-spinful example: l1=1, l2=2 (1x2) ->
    required_irreps_full=1+2+3, required_irreps=1+2+3, required_irreps_x1=None
    
    spinful example: l1=1, l2=2 (1x0.5)x(2x0.5) ->
    required_irreps_full = 1+2+3 + 0+1+2 + 1+2+3 + 2+3+4
    required_irreps = (1+2+3)x0 = 1+2+3
    required_irreps_x1 = (1+2+3)x1 = [0+1+2, 1+2+3, 2+3+4]
    
    notice that required_irreps_x1 is a list of Irreps
    '''
    p = 1
    if not no_parity:
        p = (-1) ** (l1 + l2)
    required_ls = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = Irreps([(mul, (l, p)) for l in required_ls])

    return required_irreps


def orbital_analysis(atom_orbitals, targets=None, no_parity=True): #note that atom_orbitals represent the unique elements in the structure, not the actual number of atoms 
    r'''
    example of atom_orbitals: {'42': [0, 0, 0, 1, 1, 2, 2], '16': [0, 0, 1, 1, 2]} 
    
    required_block_type: 's' - specify; 'a' - all; 'o' - off-diagonal; 'd' - diagonal; 
    '''
    hoppings_list = [] # [{'42 16': [4, 3]}, ...]
    for atom1, orbitals1 in atom_orbitals.items():
        for atom2, orbitals2 in atom_orbitals.items():
            hopping_key = atom1 + ' ' + atom2
            # if element_pairs:
            #     if hopping_key not in element_pairs:
            #         continue
            for orbital1 in range(len(orbitals1)):
                for orbital2 in range(len(orbitals2)):
                    hopping_orbital = [orbital1, orbital2]
                    hoppings_list.append({hopping_key: hopping_orbital}) #keys of hopping_list contain the atomic numbers of the two atoms, values contain the orbital indices of the two interacting orbitals 
    
    il_list = [] # [[1, 1, 2, 0], ...] this means the hopping is from 1st l=1 orbital to 0th l=2 orbital.
    for hopping in hoppings_list:
        for N_M_str, block in hopping.items():
            atom1, atom2 = N_M_str.split()
            l1 = atom_orbitals[atom1][block[0]] #finds the l of each orbital pair dictionary in the list, block contains the pair of orbital indices, which are mapped to ls by atomic_orbitals
            l2 = atom_orbitals[atom2][block[1]]
            il1 = block[0] - atom_orbitals[atom1].index(l1) #il1 specifies the index of orbitals with the same l e.g. if orbital index is 3, but it is the 2nd l=1 orbital, it is 3-1 = 2, with 1 being the index of the first l=1 orbital
            il2 = block[1] - atom_orbitals[atom2].index(l2)
        il_list.append([l1, il1, l2, il2])

    hoppings_list_mask = [False for _ in range(len(hoppings_list))] # if that hopping is already included, then it is True
    targets = []
    net_out_irreps_list = []
    net_out_irreps = Irreps(None)

    # print(hoppings_list)
    for hopping1_index in range(len(hoppings_list)):
        target = {}
        if not hoppings_list_mask[hopping1_index]: #make sure that there is no repeat in entries
            hoppings_list_mask[hopping1_index] = True
            target.update(hoppings_list[hopping1_index]) #add the key (atomic species of the interacting atoms) and its values (orbital indices of the interacting orbitals) to the target dictionary. 
            for hopping2_index in range(len(hoppings_list)):
                if not hoppings_list_mask[hopping2_index]:
                    if il_list[hopping1_index] == il_list[hopping2_index]: # il1 = il2 means the two hoppings are similar (similar means that they have the same l1 and l2 and the same orbital indices among orbitals of the same l1 and l2 )
                        target.update(hoppings_list[hopping2_index]) #if the two are similar, the target dictionary should now contain an additional entry (atomic numbers + orbital indices of interacting orbitals).
                        hoppings_list_mask[hopping2_index] = True
            targets.append(target) #each target in targets represent a specific group of similar orbital interactions, between the nth l1 orbital of atom 1 and the mth l2 orbital of atom 2

            l1, l2 = il_list[hopping1_index][0], il_list[hopping1_index][2]
            irreps_new = irreps_from_l1l2(l1, l2, 1, no_parity=no_parity)


            net_out_irreps = net_out_irreps+irreps_new
    
    return targets, net_out_irreps, net_out_irreps.sort()[0].simplify()


def process_targets(orbital_types, index_to_Z, targets): 
    Z_to_index = torch.full((100,), -1, dtype=torch.int64)
    Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))
    
    orbital_types = list(map(lambda x: np.array(x, dtype=np.int32), orbital_types))
    orbital_types_cumsum = list(map(lambda x: np.concatenate([np.zeros(1, dtype=np.int32), 
                                                                np.cumsum(2 * x + 1)]), orbital_types))

    # = process the orbital indices into block slices =
    equivariant_blocks, out_js_list = [], []
    out_slices = [0]
    for target in targets:
        out_js = None
        equivariant_block = dict()
        for N_M_str, block_indices in target.items():
            i, j = map(lambda x: Z_to_index[int(x)], N_M_str.split())
            block_slice = [
                orbital_types_cumsum[i][block_indices[0]], #defines the indices that indicate the start and end of the matrix block in row and column direction 
                orbital_types_cumsum[i][block_indices[0] + 1],
                orbital_types_cumsum[j][block_indices[1]],
                orbital_types_cumsum[j][block_indices[1] + 1]
            ]
            equivariant_block.update({N_M_str: block_slice})
            if out_js is None:
                out_js = (orbital_types[i][block_indices[0]], orbital_types[j][block_indices[1]])
            else:
                assert out_js == (orbital_types[i][block_indices[0]], orbital_types[j][block_indices[1]])
        equivariant_blocks.append(equivariant_block)
        out_js_list.append(tuple(map(int, out_js)))
        out_slices.append(out_slices[-1] + (2 * out_js[0] + 1) * (2 * out_js[1] + 1))
    
    return equivariant_blocks, out_js_list, out_slices


# Borrowed from DeepH-E3 (https://github.com/Xiaoxun-Gong/DeepH-E3.git)
class e3TensorDecomp: 
    #   module that converts between coupled and uncoupled space using Clebsch Gordan coefficients 
    def __init__(self, net_irreps_out, out_js_list, default_dtype_torch, spinful=False, no_parity=False, if_sort=False, device_torch='cpu'):
        # if spinful:
        #     default_dtype_torch = flt2cplx(default_dtype_torch)
        self.dtype = default_dtype_torch
        self.spinful = spinful
        
        self.device = device_torch
        self.out_js_list = out_js_list
        if net_irreps_out is not None:
            net_irreps_out = Irreps(net_irreps_out)

        required_irreps_out = Irreps(None)
        in_slices = [0]
        wms = []                                            # wm = wigner_multiplier
        H_slices = [0]
        wms_H = []
        for H_l1, H_l2 in out_js_list:                      # for each l1 and l2 of required H blocks in the list 
            
            # = construct required_irreps_out =
            mul = 1
            required_irreps_out_single = irreps_from_l1l2(H_l1, H_l2, mul, no_parity=no_parity)
            required_irreps_out += required_irreps_out_single
            
            # = construct slices =
            in_slices.append(required_irreps_out.dim)       # in_slices represent the orbital interaction 
            H_slices.append(H_slices[-1] + (2 * H_l1 + 1) * (2 * H_l2 + 1)) # create matrix to store the reconstructed Hamiltonian blocks 

            # = get CG coefficients multiplier to act on net_out =
            wm = []
            wm_H = []
            for _a, ir in required_irreps_out_single:
                for _b in range(mul):
                    # about this 2l+1: 
                    # we want the exact inverse of the w_3j symbol, i.e. torch.einsum("ijk,jkl->il",w_3j(l,l1,l2),w_3j(l1,l2,l))==torch.eye(...). but this is not the case, since the CG coefficients are unitary and w_3j differ from CG coefficients by a constant factor. but we know from https://en.wikipedia.org/wiki/3-j_symbol#Mathematical_relation_to_Clebsch%E2%80%93Gordan_coefficients that 2l+1 is exactly the factor we want.
                    wm.append(wigner_3j(H_l1, H_l2, ir.l, dtype=default_dtype_torch, device=device_torch))
                    wm_H.append(wigner_3j(ir.l, H_l1, H_l2, dtype=default_dtype_torch, device=device_torch) * (2 * ir.l + 1))
                    # wm.append(wigner_3j(H_l1, H_l2, ir.l, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir.l + 1))
                    # wm_H.append(wigner_3j(ir.l, H_l1, H_l2, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir.l + 1))
            wm = torch.cat(wm, dim=-1)
            wm_H = torch.cat(wm_H, dim=0)
            wms.append(wm)
            wms_H.append(wm_H)

            
        # = check net irreps out =
        # if spinful:
        #     required_irreps_out = required_irreps_out + required_irreps_out
        if net_irreps_out is not None:
            if if_sort:
                assert net_irreps_out == required_irreps_out.sort().irreps.simplify(), f'requires {required_irreps_out.sort().irreps.simplify()} but got {net_irreps_out}'
            else:
                assert net_irreps_out == required_irreps_out, f'requires {required_irreps_out} but got {net_irreps_out}'
        
        self.in_slices = in_slices
        self.wms = wms
        self.H_slices = H_slices
        self.wms_H = wms_H
        
        self.sort = None
        # if if_sort:
        #     self.sort = sort_irreps(required_irreps_out)   
        
        if self.sort is not None:
            self.required_irreps_out = self.sort.irreps_out
        else:
            self.required_irreps_out = required_irreps_out
    
    def get_H(self, net_out):
        r''' get openmx type H from net output '''
        if self.sort is not None:
            net_out = self.sort.inverse(net_out)
        out = []
        for i in range(len(self.out_js_list)):
            in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
            net_out_block = net_out[:, in_slice] # 25D output edge features are sliced according to dimension of each output l to get the right size for each required H
            H_block = torch.sum(self.wms[i][None, :, :, :] * net_out_block[:, None, None, :], dim=-1) # l3 converted back into l1 x l2 using Clebsch Gordan coefficients 
            out.append(H_block.reshape(net_out.shape[0], -1))
        return torch.cat(out, dim=-1) # output shape: [edge, (4 spin components,) H_flattened_concatenated]

    def get_net_out(self, H):
        r'''get net output from openmx type H'''
        out = []
        for i in range(len(self.out_js_list)):
            H_slice = slice(self.H_slices[i], self.H_slices[i + 1])
            
            l1, l2 = self.out_js_list[i]
            H_block = H[:, H_slice].reshape(-1, 2 * l1 + 1, 2 * l2 + 1)
            net_out_block = torch.sum(self.wms_H[i][None, :, :, :] * H_block[:, None, :, :], dim=(-1, -2))
            out.append(net_out_block)
        out = torch.cat(out, dim=-1)
        if self.sort is not None:
            out = self.sort(out)
        return out

    def convert_mask(self, mask):
        assert self.spinful
        num_edges = mask.shape[0]
        mask = mask.permute(0, 2, 1).reshape(num_edges, -1).repeat(1, 2)
        if self.sort is not None:
            mask = self.sort(mask)
        return mask
