""" Utilities for computing irreps."""

import numpy as np
# import e3nn
from e3nn.o3 import Irreps


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
