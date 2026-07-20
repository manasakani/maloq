# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
""" Utilities for computing irreps."""

import numpy as np
# import e3nn
from e3nn.o3 import Irreps

# Note: merge the first two functions...

def get_product_irreps(l1, l2, even_or_odd=None):
    """
    Return the irreps required to represent l1 X l2 (X = tensor product)
    """

    m = 1   # multiplicity
    p = 1   # even parity only 
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

def get_product_ls(l1, l2, kind='all'):
    """
    Return the l values required to represent l1 x l2.
    """

    l3s = range(abs(l1 - l2), l1 + l2 + 1)

    if kind == 'even':
        return [l for l in l3s if l % 2 == 0]
    elif kind == 'odd':
        return [l for l in l3s if l % 2 != 0]
    else:
        return list(l3s)

def get_subspace_remix_permutation(in_alpha: str, in_beta: str, ls_list: list) -> list:
    irreps_alpha = Irreps(in_alpha)
    irreps_beta = Irreps(in_beta)
    # irreps_out = Irreps(out)
    
    # Helper to map each irrep to its slice of indices, starting from an offset
    def build_irrep_pool(irreps, start_idx=0):
        pool = []
        current_idx = start_idx
        for mul, ir in irreps:
            dim = ir.dim
            channel_indices = list(range(current_idx, current_idx + dim))
            pool.append((ir, channel_indices))
            current_idx += dim
        return pool

    # Alpha starts at 0
    pool_alpha = build_irrep_pool(irreps_alpha, start_idx=0)
    # print("pool_alpha:", pool_alpha)
    
    # Beta starts at the total dimension of Alpha
    pool_beta = build_irrep_pool(irreps_beta, start_idx=irreps_alpha.dim)
    # print("pool_beta:", pool_beta)
    
    permutation = []
    alpha_track = 0
    beta_track = 0

    # loop over the interactions in ls_list:
    for i, l1 in enumerate(ls_list):
        for j, l2 in enumerate(ls_list):
            product_irreps = Irreps(str(get_product_irreps(l1, l2)))

            # if this is a diagonal block, iterate over product_irreps, pull the even ones from alpha and the odd ones from beta
            if i == j:
                for mul, ir in product_irreps:
                    if ir.l % 2 == 0:
                        # even irrep: pull from alpha
                        permutation.extend(pool_alpha[alpha_track][1])
                        alpha_track += 1
                    else:
                        # odd irrep: pull from beta
                        permutation.extend(pool_beta[beta_track][1])
                        beta_track += 1
            
            # if i> j, get the irreps from alpha:
            elif i < j:
                for mul, ir in product_irreps:
                    permutation.extend(pool_alpha[alpha_track][1])
                    alpha_track += 1

            # if i < j, get the irreps from beta:
            else:
                for mul, ir in product_irreps:
                    permutation.extend(pool_beta[beta_track][1])
                    beta_track += 1
        
    return permutation
    

def get_all_ls(ls_list):
    ls = []
    for li in ls_list:
        for lj in ls_list:
            ls.extend(get_product_ls(li, lj))
    return ls


def get_reduced_ls(ls_list, reduce_node_intra=False):
    ls = []
    ii_kind = 'even' if reduce_node_intra else 'all'
    for i, li in enumerate(ls_list):
        for j, lj in enumerate(ls_list):
            if i == j:
                ls.extend(get_product_ls(li, lj, ii_kind))
            elif i < j:
                ls.extend(get_product_ls(li, lj))
    return ls


def ls_to_irreps(ls):
    irreps = []
    for l in ls:
        irreps.append((1, (l, 1)))
    return Irreps(irreps)


def get_all_indices_dict(ls_list):
    ls_dict = {}
    idx = 0
    for i, li in enumerate(ls_list):
        for j, lj in enumerate(ls_list):
            product = get_product_ls(li, lj)
            for l in product:
                ls_dict[(i, j, l)] = idx
                idx += 2 * l + 1
    return ls_dict


def get_reduced_indices_dict(ls_list, reduce_node_intra=False):
    ls_dict = {}
    idx = 0
    ii_kind = 'even' if reduce_node_intra else 'all'
    for i, li in enumerate(ls_list):
        for j, lj in enumerate(ls_list):
            if i == j:
                product = get_product_ls(li, lj, ii_kind)
            elif i < j:
                product = get_product_ls(li, lj)
            else:
                continue

            for l in product:
                ls_dict[(i, j, l)] = idx
                idx += 2 * l + 1
    return ls_dict

def get_reduced_to_all_indices(ls_list, reduce_node_intra=False):
    reduced_indices = get_reduced_indices_dict(ls_list, reduce_node_intra=reduce_node_intra)
    indices = []
    for i, li in enumerate(ls_list):
        for j, lj in enumerate(ls_list):
            for l in get_product_ls(li, lj):
                length = 2 * l + 1
                if (i, j, l) in reduced_indices:
                    start = reduced_indices[(i, j, l)]
                    # indices.append(reduced_indices[(i, j, l)])
                else:
                    if i == j:
                        assert reduce_node_intra, "This irrep should be in the reduced set if reduce_node_intra is False"
                        indices.extend([-1] * length) # placeholder for the odd irreps that were removed in the reduced set
                        continue
                    elif i < j:
                        raise ValueError(f"This irrep should be in the reduced set since it's an upper-triangle interaction, but it's not: {(i, j, l)}")
                    else:
                        start = reduced_indices[(j, i, l)]
                        # indices.append(all_indices[(j, i, l)])
                # print(f"Adding indices for irrep {(i, j, l)}: start={start}, stop={start+length-1}")
                indices.extend(range(start, start + length))
    return indices

def get_parity_multiplier(ls_list, reduce_node_intra=False):
    parity_multiplier = [1] * get_all_len(ls_list)
    all_indices = get_all_indices_dict(ls_list)
    for i, li in enumerate(ls_list):
        for j, lj in enumerate(ls_list):
            if i == j and reduce_node_intra:
                for l in get_product_ls(li, lj, 'odd'):
                    start = all_indices[(i, j, l)]
                    length = 2 * l + 1
                    parity_multiplier[start:start+length] = [0] * length # zero out the odd irreps that were removed in the reduced set
            elif i > j:
                kind = 'even' if (li + lj) % 2 == 1 else 'odd' # if the outer parity is odd, we need to flip the sign of the even irreps, and vice versa
                for l in get_product_ls(li, lj, kind):
                    start = all_indices[(i, j, l)]
                    length = 2 * l + 1
                    parity_multiplier[start:start+length] = [-1] * length
                    # print(f"Flipping parity for irrep {(i, j, l)}: start={start}, stop={start+length-1}")
    return parity_multiplier


def get_all_indices(ls_list):
    return np.cumsum([0] + [2*l + 1 for l in get_all_ls(ls_list)])


def get_reduced_indices(ls_list, reduce_node_intra=False):
    return np.cumsum([0] + [2*l + 1 for l in get_reduced_ls(ls_list, reduce_node_intra=reduce_node_intra)])


def get_all_len(ls_list):
    return sum([2*l + 1 for l in get_all_ls(ls_list)])


def get_reduced_len(ls_list, reduce_node_intra=False):
    return sum([2*l + 1 for l in get_reduced_ls(ls_list, reduce_node_intra=reduce_node_intra)])


# this is here just for reference! Not used
def get_edge_permutation(self):
    """
    The forward and backward edges contain the same irreps, but they are permuted in the data list
    due to the order of flattening the matrix blocks. Here we create the permutation of the irreps to match the reverse edge order.
    We also handle the reflection rules of the orbital interactions, which are different for even and odd parity.
    """

    full_irrep_len = [sum([2*l + 1 for l in Irreps(str(get_product_irreps(l1, l2))).ls]) for l1 in self.ls_list for l2 in self.ls_list]
    edge_permutation = [0] * sum(full_irrep_len)
    self.edge_m_reflection = np.ones(sum(full_irrep_len), dtype=int)
    forward_irrep_track = {}
    pointer = 0

    total_irreps = Irreps('')

    for i, l1 in enumerate(self.ls_list):
        for j, l2 in enumerate(self.ls_list):

            # --> 1. Handle the permutation of the irreps:
            product_irreps = str(get_product_irreps(l1, l2))
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