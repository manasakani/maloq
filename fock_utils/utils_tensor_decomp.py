from e3nn.o3 import Irreps, wigner_3j
from e3nn.nn import Extract
import torch
import numpy as np

# functions in this file are adapted from: https://github.com/Xiaoxun-Gong/DeepH-E3

def l1l2_to_l3s(l1, l2):
        
    m = 1   # multiplicity
    p = 1   # even parity only (real-valued Fock matrix)
    l3s = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = Irreps([(m, (l, p)) for l in l3s])
    return required_irreps

class sort_irreps(torch.nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()
        
        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = Extract(irreps_in, irreps_out_list, instructions)
        
        irreps_in_list = [((mul, ir),) for mul, ir in irreps_in]
        instructions_inv = [(i,) for i in sorted_irreps.p]
        self.extr_inv = Extract(sorted_irreps.irreps, irreps_in_list, instructions_inv)
        
        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps.simplify()
    
    def forward(self, x):
        r'''irreps_in -> irreps_out'''
        extracted = self.extr(x)
        return torch.cat(extracted, dim=-1)

    def inverse(self, x):
        r'''irreps_out -> irreps_in'''
        extracted_inv = self.extr_inv(x)
        return torch.cat(extracted_inv, dim=-1)

class e3TensorDecomp:
    """
    Transformation between the coupled and uncoupled bases
    """
    def __init__(self, net_irreps_out, out_js_list, default_dtype_torch, if_sort=False, device_torch='cpu'):
        self.dtype = default_dtype_torch
        
        self.device = device_torch
        self.out_js_list = out_js_list
        if net_irreps_out is not None:
            net_irreps_out = Irreps(net_irreps_out)

        required_irreps_out = Irreps(None)
        in_slices = [0]
        wms = [] # wm = wigner_multiplier
        H_slices = [0]
        wms_H = []
            
        for H_l1, H_l2 in out_js_list:
            
            # = construct required_irreps_out =
            mul = 1
            required_irreps_out_single = l1l2_to_l3s(H_l1, H_l2)
            required_irreps_out += required_irreps_out_single
           
            # = construct slices =
            in_slices.append(required_irreps_out.dim)
            H_slices.append(H_slices[-1] + (2 * H_l1 + 1) * (2 * H_l2 + 1))
            
            # = get CG coefficients multiplier to act on net_out =
            wm = []
            wm_H = []
            for _a, ir in required_irreps_out_single:
                for _b in range(mul):
                    # about this 2l+1: 
                    # we want the exact inverse of the w_3j symbol, i.e. torch.einsum("ijk,jkl->il",w_3j(l,l1,l2),w_3j(l1,l2,l))==torch.eye(...). 
                    # but this is not the case, since the CG coefficients are unitary and w_3j differ from CG coefficients by a constant factor. 
                    # but we know from https://en.wikipedia.org/wiki/3-j_symbol#Mathematical_relation_to_Clebsch%E2%80%93Gordan_coefficients that 2l+1 is exactly the factor we want.
                    wm.append(wigner_3j(H_l1, H_l2, ir.l, dtype=default_dtype_torch, device=device_torch))
                    wm_H.append(wigner_3j(ir.l, H_l1, H_l2, dtype=default_dtype_torch, device=device_torch) * (2 * ir.l + 1))

            wm = torch.cat(wm, dim=-1)
            wm_H = torch.cat(wm_H, dim=0)
            wms.append(wm)
            wms_H.append(wm_H)
            
        # = check net irreps out =
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
        if if_sort:
            self.sort = sort_irreps(required_irreps_out) # TODO: check effect of sort and add the implementation here
        
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
            net_out_block = net_out[:, in_slice]
            H_block = torch.sum(self.wms[i][None, :, :, :] * net_out_block[:, None, None, :], dim=-1)
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
        

def make_output_irreps(orbital_basis):
    '''
    hoppings_list = {'atomic#1, atomic#2': [orb_idx1, orb_idx2], ...}
    il_list = [l1, idx_l1, l2, idx_l2] # hopping term from idx_l1's l1 orbital to the idx_l2's l2 orbital on the corresponding atoms in hoppings_list 
    '''

    hoppings_list = [] 
    for atom1, orbitals1 in orbital_basis.items():
        for atom2, orbitals2 in orbital_basis.items():
            hopping_key = str(atom1) + ' ' + str(atom2)
            for orbital1 in range(len(orbitals1)):
                for orbital2 in range(len(orbitals2)):
                    hopping_orbital = [orbital1, orbital2]
                    hoppings_list.append({hopping_key: hopping_orbital}) 
    
    il_list = [] 
    for hopping in hoppings_list:
        for N_M_str, block in hopping.items():
            atom1, atom2 = N_M_str.split()
            l1 = orbital_basis[int(atom1)][block[0]] 
            l2 = orbital_basis[int(atom2)][block[1]]
            il1 = block[0] - orbital_basis[int(atom1)].index(l1) 
            il2 = block[1] - orbital_basis[int(atom2)].index(l2)
        il_list.append([l1, il1, l2, il2])

    # used to exclude double-counted interactions
    hoppings_list_mask = [False for _ in range(len(hoppings_list))]
    targets = []
    net_out_irreps = Irreps(None)

    for hopping1_index in range(len(hoppings_list)):
        target = {}
        if not hoppings_list_mask[hopping1_index]: 
            hoppings_list_mask[hopping1_index] = True
            target.update(hoppings_list[hopping1_index]) 
            for hopping2_index in range(len(hoppings_list)):
                if not hoppings_list_mask[hopping2_index]:
                    if il_list[hopping1_index] == il_list[hopping2_index]:
                        target.update(hoppings_list[hopping2_index]) 
                        hoppings_list_mask[hopping2_index] = True
            targets.append(target) 

            l1, l2 = il_list[hopping1_index][0], il_list[hopping1_index][2]
            irreps_new = l1l2_to_l3s(l1, l2)

            net_out_irreps = net_out_irreps + irreps_new
    
    # each target in targets represent a specific group of similar orbital interactions, between the nth l1 orbital of atom 1 and the mth l2 orbital of atom 2
    # the number of targets is the number of orbital interaction blocks, or (Norb_1 + N_orb_2 + ...)^3 over the different atomic species 
    return targets, net_out_irreps, net_out_irreps.sort()[0].simplify()
    

def process_targets(orbital_basis, targets): 

    orbital_types = [orbital_basis[atom] for atom in orbital_basis.keys()]
    index_to_Z = list(orbital_basis.keys())

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
                            orbital_types_cumsum[i][block_indices[0]], # defines the indices that indicate the start and end of the matrix block in row and column direction 
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
