from e3nn.o3 import Irreps, wigner_3j
from e3nn.nn import Extract
import torch
import numpy as np
import re

# the e3TensorDecomp class is adapted from: https://github.com/Xiaoxun-Gong/DeepH-E3 and modified to account for diffuse functions

def l1l2_to_l3s(l1, l2):

    # handle convention for diffuse functions
    l1 = l1 % 10
    l2 = l2 % 10
        
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
            
            # Handle convention for diffuse functions
            H_l1 = H_l1 % 10
            H_l2 = H_l2 % 10
            
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
            self.sort = sort_irreps(required_irreps_out) 
        
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

            l1 = l1 % 10  # handle convention for diffuse functions
            l2 = l2 % 10
            
            H_block = H[:, H_slice].reshape(-1, 2 * l1 + 1, 2 * l2 + 1)
            net_out_block = torch.sum(self.wms_H[i][None, :, :, :] * H_block[:, None, :, :], dim=(-1, -2))
            out.append(net_out_block)

        out = torch.cat(out, dim=-1)
        
        if self.sort is not None:
            out = self.sort(out)
        return out
        

def make_output_irreps_old(orbital_basis):
    '''
    hoppings_list = {'atomic#1, atomic#2': [orb_idx1, orb_idx2], ...}
    il_list = [l1, idx_l1, l2, idx_l2] # hopping term from idx_l1's l1 orbital to the idx_l2's l2 orbital on the corresponding atoms in hoppings_list 
    '''

    # add 10 to all diffuse orbitals to distinguish them from core orbitals
    def find_diffuse_start(orbitals):
        for i in range(1, len(orbitals)):
            if orbitals[i] < orbitals[i-1]:
                return i
        return len(orbitals)  

    for atom1, orbitals in orbital_basis.items():
        diffuse_start = find_diffuse_start(orbitals)
        orbitals[diffuse_start:] = [l + 10 for l in orbitals[diffuse_start:]]

    # Collect all the orbital interactions for every possible pair of atoms in the orbital basis:
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
    

def make_output_irreps(orbital_basis):
    '''
    hoppings_list = {'atomic#1, atomic#2': [orb_idx1, orb_idx2], ...}
    il_list = [l1, idx_l1, l2, idx_l2] # hopping term from idx_l1's l1 orbital to the idx_l2's l2 orbital on the corresponding atoms in hoppings_list 
    '''

    orbital_type_dict = {0: 's_', 1: 'p_', 2: 'd_', 3: 'f_', 4: 'g_', 10: 'sd_', 11: 'pd_', 12: 'dd_'} # the last are diffuse functions

    # add 10 to any diffuse orbitals to distinguish them from core orbitals
    def find_diffuse_start(orbitals):
        for i in range(1, len(orbitals)):
            if orbitals[i] < orbitals[i-1]:
                return i
        return len(orbitals)  

    for atom1, orbitals in orbital_basis.items():
        diffuse_start = find_diffuse_start(orbitals)
        orbitals[diffuse_start:] = [l + 10 for l in orbitals[diffuse_start:]]

    # Find the maximum target size needed to accommodate all orbital interactions
    ls_list = []
    for l in range(20): # large to account for possible diffuse functions which are incremented by 10
        counts = [torch.sum(torch.tensor(orbital_basis[el]) == l) for el in orbital_basis]
        max_count = max(counts).item() 
        ls_list.append(torch.tensor(max_count * [l], dtype=torch.int))          
    ls_list = torch.cat(ls_list).tolist()        # Ex: [5s, 4p, 3d, 0f, 0g] - ls_list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2].
    # print("ls_list: ", ls_list)                  # for OMOL: tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 10, 11, 12]

    # Compute the full direct sum of irreps required to describe this set of orbital interactions:
    req_output_irreps = Irreps('')
    out_slices = [0]
    for i, l1 in enumerate(ls_list):
        for j, l2 in enumerate(ls_list):
            product_irreps = l1l2_to_l3s(l1, l2)
            irrep_len = sum([2*(l%10) + 1 for l in Irreps(product_irreps).ls])
            req_output_irreps += Irreps(product_irreps)
            out_slices.append(np.int32(out_slices[-1] + irrep_len))

    # We make a dict of the form 'atom#1 atom#1': s0-p1', etc
    atomic_interactions = {}
    # # Iterate through the orbital interactions of every atom:
    for atom1, orbitals1 in orbital_basis.items():
        for atom2, orbitals2 in orbital_basis.items():
            atom_interaction_key = str(atom1) + ' ' + str(atom2)

            orbital_interaction_keys = []
            for i, orbital1 in enumerate(orbitals1):
                orbital_type_string = orbital_type_dict[orbital1]  
                orbital_multiplicity = sum(1 for j in range(i) if orbitals1[j] == orbital1)  # Count how many orbitals of the same l value come before this
                orbital1_type = orbital_type_string + str(orbital_multiplicity)  # e.g. 's0', 'p1', 'd2', etc.
                
                for k, orbital2 in enumerate(orbitals2):
                    orbital_type_string = orbital_type_dict[orbital2]
                    orbital_multiplicity = sum(1 for j in range(k) if orbitals2[j] == orbital2)  # Count how many orbitals of the same l value come before this
                    orbital2_type = orbital_type_string + str(orbital_multiplicity)  
                    orbital_interaction_keys.append(orbital1_type + '-' + orbital2_type)        # e.g. 's0-p1', 'p1-d2', etc.

            atomic_interactions.update({atom_interaction_key: orbital_interaction_keys})
    # print("atomic_interactions: ", atomic_interactions)

    # This will contain the specific orbital interaction (eg 's0-p1') for every block of the target
    full_orb_interaction_list = []
    for i, orb1 in enumerate(ls_list):
        orbital1_type = orbital_type_dict[orb1]  
        orbital_multiplicity1 = sum(1 for j in range(i) if ls_list[j] == orb1)
        orbital1_type = orbital1_type + str(orbital_multiplicity1)
        for k, orb2 in enumerate(ls_list):
            orbital2_type = orbital_type_dict[orb2]  
            orbital_multiplicity2 = sum(1 for j in range(k) if ls_list[j] == orb2)
            orbital2_type = orbital2_type + str(orbital_multiplicity2)
            full_orb_interaction_list.append(orbital1_type + '-' + orbital2_type)  # e.g. 's0-p1', 'p1-d2', etc.
    # print("full_orb_interaction_list: ", full_orb_interaction_list)

    # Now, for every block in full_orb_interaction_list, find all the atomic interaction keys that contain this block
    # out_js just contains tuples which describes the l-orbital interaction for every target block (eg, [(0, 0), (0, 1) ...])
    len_ls_list = len(ls_list)
    targets = []
    out_js_list = []
    for block_ind, block in enumerate(full_orb_interaction_list):
        target = {}
        for atom_interaction_key, orbital_interaction_keys in atomic_interactions.items():
            if block in orbital_interaction_keys:
                # Find the indices of the orbitals in the ls_list (row and column of the targets matrix)
                idx1 = orbital_interaction_keys.index(block) // len_ls_list # row
                idx2 = orbital_interaction_keys.index(block) % len_ls_list # col
                
                N_M_str = f"{atom_interaction_key}"
                target.update({N_M_str: [idx1, idx2]})

        out_js = (ls_list[block_ind // len_ls_list], ls_list[block_ind % len_ls_list])
        out_js_list.append(out_js)

        if target:
            targets.append(target)

    # print("targets: ", targets)
    # print("out_js_list: ", out_js_list)
    # Each element of 'targets' is a set of orbital interactions between different atoms, which can be inserted into that targer.
    # each target in targets represent a specific group of similar orbital interactions, between the nth l1 orbital of atom 1 and the mth l2 orbital of atom 2
    # the number of targets is the number of orbital interaction blocks, or (Norb_1 + N_orb_2 + ...)^3 over the different atomic species 
    return targets, req_output_irreps, req_output_irreps.sort()[0].simplify(), ls_list, out_js_list, out_slices, full_orb_interaction_list
    

def process_targets(orbital_basis, targets, ls_list=None, out_js_list=None, full_orb_interaction_list=None): 

    orbital_type_dict = {0: 's_', 1: 'p_', 2: 'd_', 3: 'f_', 4: 'g_', 10: 'sd_', 11: 'pd_', 12: 'dd_'}
    reverse_orbital_type_dict = {v: k for k, v in orbital_type_dict.items()}

    orbital_types = [orbital_basis[atom] for atom in orbital_basis.keys()]
    index_to_Z = list(orbital_basis.keys())

    Z_to_index = torch.full((100,), -1, dtype=torch.int64)
    Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))

    # Record where each orbital block starts in the corresponding atom-pair's interaction matrix
    equivariant_blocks = []
    out_slices = [0]
    for target_ind, target in enumerate(targets):

        # l1, l2 = out_js_list[target_ind]  # the interaction is defined by the out_js, which gives us the l1 and l2
        target_orb_interaction = full_orb_interaction_list[target_ind].split('-')

        equivariant_block = dict()
        for N_M_str, block_indices in target.items():
            i, j = map(lambda x: Z_to_index[int(x)], N_M_str.split())

            # now we need to find the block of the matrix that corresponds to the interaction between
            # the target_orb_interaction[0] orbital of atom i and the target_orb_interaction[1] orbital of atom j

            # reverse orbital type dict needs 's', 'p' .. etc but we have 's0', 'p1', 'd2' etc.
            parts1 = re.split(r'(\d+)', target_orb_interaction[0])
            parts2 = re.split(r'(\d+)', target_orb_interaction[1])
            l1_type = parts1[0]  # orbital type (e.g., 's_', 'p_')
            l1_level = int(parts1[1])  # multiplicity number
            l2_type = parts2[0]  # orbital type (e.g., 's_', 'p_')  
            l2_level = int(parts2[1])  # multiplicity number

            atom1_basis = np.array(orbital_types[i], dtype=np.int32)
            atom2_basis = np.array(orbital_types[j], dtype=np.int32)
            
            # find the index of l1_type, l1_level in the atom1_basis
            l1_indices = np.where(atom1_basis == reverse_orbital_type_dict[l1_type])[0]
            l1_index = l1_indices[l1_level]  # Use level as index into matching orbitals
            
            # find the index of l2_type, l2_level in the atom2_basis
            l2_indices = np.where(atom2_basis == reverse_orbital_type_dict[l2_type])[0]
            l2_index = l2_indices[l2_level]  # Use level as index into matching orbitals

            # --> Calculate the start and end positions of the row/column in the interaction matrix for this orbital pair
            if l1_index == 0:
                atom1_start_row = 0
            else:
                atom1_start_row = np.cumsum([2 * (atom1_basis[:l1_index] % 10) + 1])[-1]
            atom1_end_row = atom1_start_row + (2 * (atom1_basis[l1_index] % 10) + 1)
            
            if l2_index == 0:
                atom2_start_col = 0
            else:
                atom2_start_col = np.cumsum([2 * (atom2_basis[:l2_index] % 10) + 1])[-1]
            atom2_end_col = atom2_start_col + (2 * (atom2_basis[l2_index] % 10) + 1)

            # --> Record the start/end row/col into the block slice for this 'atom 1 atom2' for this target.
            block_slice = [int(atom1_start_row), int(atom1_end_row), int(atom2_start_col), int(atom2_end_col)]

            equivariant_block.update({N_M_str: block_slice})
    
        equivariant_blocks.append(equivariant_block)

    return equivariant_blocks

def process_targets_old(orbital_basis, targets): 

    orbital_types = [orbital_basis[atom] for atom in orbital_basis.keys()]
    index_to_Z = list(orbital_basis.keys())

    Z_to_index = torch.full((100,), -1, dtype=torch.int64)
    Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))
        
    orbital_types = list(map(lambda x: np.array(x, dtype=np.int32), orbital_types))
    orbital_types_cumsum = list(map(lambda x: np.concatenate([np.zeros(1, dtype=np.int32), 
                                                                np.cumsum(2 * (x % 10) + 1)]), orbital_types))

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
        out_slices.append(out_slices[-1] + (2 * (out_js[0]%10) + 1) * (2 * (out_js[1]%10) + 1))
    
    return equivariant_blocks, out_js_list, out_slices
