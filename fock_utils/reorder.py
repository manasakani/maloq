import numpy as np
from itertools import product
from collections import deque
import pymetis


def get_num_remote_nodes(rank, node_counts, node_displacements, edge_index):
    """Calculates the number of remote nodes in a partition.

    Parameters
    ----------
    rank : int
        rank of the partition
    node_counts : NDArray
        number of nodes in each partition
    node_displacements : NDArray
        displacement of the nodes in each partition
    edge_index : NDArray
        edge index of the graph

    Returns
    -----------
    num_remote_nodes : int
        number of remote nodes in the partition

    """

    # start and end nodes on every rank:
    start_nodes = node_displacements
    end_nodes = node_displacements + node_counts

    #  get 'remote' nodes in this rank to be recieved from remote ranks
    mask = (edge_index < node_counts[rank]) | (
        edge_index >= node_counts[rank] + node_displacements[rank]
    )
    filtered_nodes = edge_index[mask]

    remote_nodes = []
    for start, end in zip(start_nodes, end_nodes):
        range_mask = (filtered_nodes >= start) & (filtered_nodes < end)
        matching_nodes = filtered_nodes[range_mask]

        # Append only unique nodes from the matching ones
        remote_nodes.extend(matching_nodes)

    # Remove duplicates (ensure unique remote nodes)
    remote_nodes = np.unique(remote_nodes)

    return len(remote_nodes)


def get_comm_volume(adj_matrix, partition):
    """Computes the communication volume between the partitions.

    Parameters
    ----------
    adj_matrix : NDArray
        adjacency matrix of the atomic structure
    partition : list
        list of atom indices in the partition

    Returns
    -----------
    comm_volume : int
        total communication volume between the partitions

    """

    num_ranks = len(partition)

    # reorder/reduce the matrix
    all_indices_partition = np.concatenate(
        [par.reshape(-1) for par in partition], axis=-1
    )
    reduced_adj_matrix = adj_matrix.tocsr()[all_indices_partition, :][
        :, all_indices_partition
    ].tocoo()
    col_indices = reduced_adj_matrix.col

    counts = np.array([len(partition[i]) for i in range(num_ranks)])
    displacements = np.cumsum(counts) - counts[0]

    # volume per parition
    comm_volume = []
    for i in range(num_ranks):
        comm_volume.append(get_num_remote_nodes(i, counts, displacements, col_indices))

    return np.sum(comm_volume)


def get_num_neighbors(adj_matrix, partition):
    """Computes the number of neighbors between the partitions.

    Parameters
    ----------
    adj_matrix : NDArray
        adjacency matrix of the atomic structure
    partition : list
        list of atom indices in the partition

    Returns
    -----------
    neighbors : int
        total number of neighbors between the partitions

    """

    neighbors = 0

    adj_matrix = adj_matrix.tocsr()

    for i, par1 in enumerate(partition):
        for j, par2 in enumerate(partition):
            if i == j:
                continue
            else:
                piece = adj_matrix[par1, :][:, par2]
                if piece.nnz > 0:
                    neighbors += 1
                # piece = adj_matrix[par1, :][:, par2]
                # if np.sum(piece) > 0:
                #     neighbors += 1

    return neighbors


def get_cut_position(
    atom_positions, atom_degree, atom_indices, cut_dim, cell_size, origin=None
):
    """Gets thhe position to cut the domain based equal atom_degrees in the subdomains.

    Parameters
    ----------
    atom_positions : NDArray
        position of all the atoms
    atom_degree : NDArray
        atom_degrees of all the atoms
    atom_indices : NDArray
        indices of the atoms
    cut_dim : int
        dimension to cut
    cell_size : NDArray
        size of the cell
    origin : NDArray, optional
        origin of the cut

    Returns
    -----------
    cut_position : float
        position to cut
    left_indices : NDArray
        indices of the left partition
    right_indices : NDArray
        indices of the right partition

    """

    # Unwrap all the atoms in the largest dimension
    temp_atom_pos = atom_positions.copy()
    if origin is not None:
        for p in temp_atom_pos:
            if p[cut_dim] < origin[cut_dim]:
                p[cut_dim] += cell_size[cut_dim]

    # atom indices is the local set of atom indices in the subdomain
    sorted_indices = atom_indices[np.argsort(temp_atom_pos[atom_indices, cut_dim])]

    # decide where to cut based on the median value of the atom_degree
    sorted_degree = atom_degree[sorted_indices]
    degree_cumsum = np.cumsum(sorted_degree)
    total_degree = degree_cumsum[-1]
    split_idx = np.searchsorted(degree_cumsum, total_degree / 2)
    cut_position = temp_atom_pos[sorted_indices][split_idx, cut_dim]

    # split atoms into left and right groups based on the median value
    left_indices = sorted_indices[temp_atom_pos[sorted_indices, cut_dim] < cut_position]
    right_indices = sorted_indices[
        temp_atom_pos[sorted_indices, cut_dim] >= cut_position
    ]

    if origin is not None:
        if cut_position > cell_size[cut_dim]:
            cut_position -= cell_size[cut_dim]

    return cut_position, left_indices, right_indices


def get_surface_volume(sub_domain_size, periodicity, rcut):
    """Get the surface volume of the subdomain.

    Parameters
    ----------
    sub_domain_size : NDArray
        size of the subdomain
    periodicity : NDArray
        periodicity of the subdomain
    rcut : float
        cutoff radius

    Returns
    -----------
    surface_volume : float
        surface volume of the subdomain

    """

    surface_volume = 0.0

    # loop over surfaces
    for l in range(3):

        # get the other two dimensions that are not l
        remaining = {0, 1, 2} - {l}
        j, k = list(remaining)

        # if periodic then neighbor is oneself
        if not periodicity[l]:
            # 2x since there are two opposite surfaces
            surface_volume += 2 * rcut * sub_domain_size[j] * sub_domain_size[k]

    # take edges into account
    for l in range(3):
        for j in range(3):
            if l != j:
                if not (periodicity[l] and periodicity[j]):
                    k = 3 - l - j
                    # / 2 since there are two opposite edges
                    surface_volume += np.pi * rcut**2 / 2 * sub_domain_size[k]

    # take corners into account
    if not (periodicity[0] and periodicity[1] and periodicity[2]):
        surface_volume += 4 / 3 * np.pi * rcut**3

    return surface_volume


def partition_surface_volume(
    levels,
    atom_positions,
    atom_degrees,
    cell_size,
    output_partition,
    rcut,
    is_periodic=[True, True, True],
    atom_indices=None,
):
    """Partitions the domain based on the surface volume.

    Parameters
    ----------
    levels : int
        number of levels to partition
    atom_positions : NDArray
        position of all the atoms
    atom_degrees : NDArray
        atom_degrees of all the atoms
    cell_size : NDArray
        size of the cell
    output_partition : list
        Resulting output_partition
    rcut : float
        cutoff radius
    is_periodic : list, optional
        periodicity of the domain
    atom_indices : NDArray, optional
        indices of the atoms

    """

    if atom_indices is None:
        atom_indices = np.arange(atom_positions.shape[0])

    if levels == 0:
        output_partition.append(atom_indices)
        return

    # Get the size of the subdomain
    # Assume that the subdomain is a cube and contiguous
    lx = np.max(atom_positions[atom_indices][:, 0]) - np.min(
        atom_positions[atom_indices][:, 0]
    )
    ly = np.max(atom_positions[atom_indices][:, 1]) - np.min(
        atom_positions[atom_indices][:, 1]
    )
    lz = np.max(atom_positions[atom_indices][:, 2]) - np.min(
        atom_positions[atom_indices][:, 2]
    )
    sub_domain_size = np.array([lx, ly, lz])

    # find the surface volume after each possible cut
    surface_volume = np.zeros(3)
    for i in range(3):
        new_sub_domain_size = sub_domain_size.copy()
        # Half the size of the subdomain in this dimension
        new_sub_domain_size[i] = sub_domain_size[i] / 2
        # not periodic anymore after cut
        new_periodicity = is_periodic.copy()
        new_periodicity[i] = False

        surface_volume[i] = get_surface_volume(
            new_sub_domain_size, new_periodicity, rcut
        )

    largest_dim = np.argmin(surface_volume)
    is_periodic = is_periodic.copy()
    is_periodic[largest_dim] = False

    # 2. Calculate the cut index in the current dimension
    _, left_indices, right_indices = get_cut_position(
        atom_positions, atom_degrees, atom_indices, largest_dim, cell_size
    )

    # Recursively cut the domain
    partition_surface_volume(
        levels - 1,
        atom_positions,
        atom_degrees,
        cell_size,
        output_partition,
        rcut,
        is_periodic,
        atom_indices=left_indices,
    )
    partition_surface_volume(
        levels - 1,
        atom_positions,
        atom_degrees,
        cell_size,
        output_partition,
        rcut,
        is_periodic,
        atom_indices=right_indices,
    )


def get_neighbors(sub_domain_size, periodicity, rcut):

    num_neighbors = 0

    # loop over surfaces
    for l in range(3):

        # if periodic then neighbor is oneself
        if not periodicity[l]:
            # 2x since there are two opposite surfaces
            num_neighbors += 2 * np.ceil(rcut / sub_domain_size[l])

    # take edges into account
    for l in range(3):
        for j in range(3):
            if l != j:
                if not (periodicity[l] or periodicity[j]):
                    diagonal = np.sqrt(
                        sub_domain_size[l] ** 2 + sub_domain_size[j] ** 2
                    )
                    # / 2 since there are two opposite edges
                    num_neighbors += 2 * np.ceil(rcut / diagonal)

    # take corners into account
    if not (periodicity[0] or periodicity[1] or periodicity[2]):

        diagonal = np.sqrt(
            sub_domain_size[0] ** 2 + sub_domain_size[1] ** 2 + sub_domain_size[2] ** 2
        )
        num_neighbors += 8 * np.ceil(rcut / diagonal)

    return num_neighbors


def partition_approx_neighbors(
    levels,
    atom_positions,
    atom_degrees,
    cell_size,
    output_partition,
    rcut,
    is_periodic=[True, True, True],
    atom_indices=None,
):
    """Partitions the domain based on the number of neighbors.

    Parameters
    ----------
    levels : int
        number of levels to partition
    atom_positions : NDArray
        position of all the atoms
    atom_degrees : NDArray
        atom_degrees of all the atoms
    cell_size : NDArray
        size of the cell
    output_partition : list
        Resulting output_partition
    rcut : float
        cutoff radius
    is_periodic : list, optional
        periodicity of the domain
    atom_indices : NDArray, optional
        indices of the atoms
    """

    if atom_indices is None:
        atom_indices = np.arange(len(atom_positions))

    if levels == 0:
        output_partition.append(atom_indices)
        return

    # Get the size of the subdomain
    # Assume that the subdomain is a cube and contiguous
    lx = np.max(atom_positions[atom_indices][:, 0]) - np.min(
        atom_positions[atom_indices][:, 0]
    )
    ly = np.max(atom_positions[atom_indices][:, 1]) - np.min(
        atom_positions[atom_indices][:, 1]
    )
    lz = np.max(atom_positions[atom_indices][:, 2]) - np.min(
        atom_positions[atom_indices][:, 2]
    )
    sub_domain_size = np.array([lx, ly, lz])

    # --> Find the largest dimension to cut
    if np.all(is_periodic):
        largest_dim = np.argmax(sub_domain_size)
    else:
        num_new_neighbors = np.zeros(
            3
        )  # the number of neighbors created if we split in this dimension
        for i in range(3):
            new_sub_domain_size = sub_domain_size.copy()
            # Half the size of the subdomain in this dimension
            new_sub_domain_size[i] = sub_domain_size[i] / 2
            # not periodic anymore after cut
            new_periodicity = is_periodic.copy()
            new_periodicity[i] = False

            num_new_neighbors[i] = get_neighbors(
                new_sub_domain_size, new_periodicity, rcut
            )

        largest_dim = np.argmin(num_new_neighbors)

    is_periodic = is_periodic.copy()
    is_periodic[largest_dim] = False

    _, left_indices, right_indices = get_cut_position(
        atom_positions, atom_degrees, atom_indices, largest_dim, cell_size
    )

    partition_approx_neighbors(
        levels - 1,
        atom_positions,
        atom_degrees,
        cell_size,
        output_partition,
        rcut,
        is_periodic,
        atom_indices=left_indices,
    )
    partition_approx_neighbors(
        levels - 1,
        atom_positions,
        atom_degrees,
        cell_size,
        output_partition,
        rcut,
        is_periodic,
        atom_indices=right_indices,
    )


def partition_old_neighbors(
    levels,
    atom_positions,
    atom_degrees,
    cell_size,
    output_partition,
    rcut,
    cuts=[0, 0, 0],
    atom_indices=None,
    origin=None,
):
    """Partitions the domain based on the number of neighbors.

    Parameters
    ----------
    levels : int
        number of levels to partition
    atom_positions : NDArray
        position of all the atoms
    atom_degrees : NDArray
        atom_degrees of all the atoms
    cell_size : NDArray
        size of the cell
    output_partition : list
        Resulting output_partition
    rcut : float
        cutoff radius
    cuts : list, optional
        # cuts in this dimension of the domain (<2 = periodic)
    atom_indices : NDArray, optional
        indices of the atoms
    origin : NDArray, optional
        origin of the cut
    """

    if atom_indices is None:
        atom_indices = np.arange(len(atom_positions))

    if origin is None:
        origin = np.zeros_like(sub_domain_size, dtype=float)

    # wrap around position for size
    temp_atom_pos = atom_positions.copy()
    for p in temp_atom_pos:
        for i in range(3):
            if p[i] < origin[i]:
                p[i] += cell_size[i]

    # --> Get the size of the subdomain
    lx = np.max(temp_atom_pos[atom_indices][:, 0]) - np.min(temp_atom_pos[atom_indices][:, 0])
    ly = np.max(temp_atom_pos[atom_indices][:, 1]) - np.min(temp_atom_pos[atom_indices][:, 1])
    lz = np.max(temp_atom_pos[atom_indices][:, 2]) - np.min(temp_atom_pos[atom_indices][:, 2])
    sub_domain_size = np.array([lx, ly, lz])

    if levels == 0:
        output_partition.append(atom_indices)
        return


    # --> This is the case where the domain has not yet been split due to pbc, we cut again
    if any([p == 0 for p in cuts]):
        smallest_indices = np.where([p == 0 for p in cuts])[0]
        pick = np.argmax(sub_domain_size[smallest_indices])                     # break ties with the largest dimension
        dim_to_cut = smallest_indices[pick]

        # Calculate the cut index in the current dimension (left and right indices are not used because this is a periodic cut)
        cut_position, left_indices, right_indices = get_cut_position(atom_positions, atom_degrees, atom_indices, dim_to_cut, sub_domain_size, origin=origin)
        
        # Update the origin to the cut location
        origin = origin.copy()
        origin[dim_to_cut] = cut_position

        # Cut periodicity from this dimension, so the next cut will make a new piece
        cuts_now = cuts.copy()
        cuts_now[dim_to_cut] += 1

        partition_old_neighbors(
            levels,
            atom_positions,
            atom_degrees,
            cell_size,
            output_partition,
            rcut,
            cuts_now,
            atom_indices=atom_indices,
            origin=origin,
        )

    # the periodicity has been cut in all dimensions, we can now split the domain
    else:

        num_dim_neighbors = np.zeros(3)             # the number of neighbors created if we split in this dimension
        for i in range(3):
            if cuts[i] == 1:                        # if the domain hasn't been split yet, splitting it creates 1 neighbor for each piece
                num_dim_neighbors[i] = 1
            else:
                # num_dim_neighbors[i] = 2 * np.ceil( rcut / sub_domain_size[i] )
                num_dim_neighbors[i] = np.ceil( 2 * rcut / sub_domain_size[i] )
                    
            smallest_indices = np.where(num_dim_neighbors == np.min(num_dim_neighbors))[0]
            
            if len(smallest_indices) == 1:
                dim_to_cut = np.argmin(num_dim_neighbors) 
            if len(smallest_indices) > 1:
                pick = np.argmax(sub_domain_size[smallest_indices]) # break ties with the largest dimension
                dim_to_cut = smallest_indices[pick]

        # --> Calculate the cut index in the current dimension
        cut_position, left_indices, right_indices = get_cut_position(atom_positions, atom_degrees, atom_indices, dim_to_cut, sub_domain_size, origin=origin)
        # if len(left_indices) == 0 or len(right_indices) == 0:
        #     print("Warning: Empty partition detected! Stopping recursion.")
        #     return

        # Update the origin and size of the largest dimension for the next cut
        sub_domain_size = sub_domain_size.copy()
        sub_domain_size[dim_to_cut] = sub_domain_size[dim_to_cut] / 2

        # update the cuts:
        cuts_now = cuts.copy()
        cuts_now[dim_to_cut] += 1
        
        # Recursively cut the domain
        origin_left = origin.copy()
        origin_right = origin.copy()
        origin_right[dim_to_cut] = cut_position

        partition_old_neighbors(
            levels - 1,
            atom_positions,
            atom_degrees,
            cell_size,
            output_partition,
            rcut,
            cuts_now,
            atom_indices=left_indices,
            origin=origin_left,
        )
        partition_old_neighbors(
            levels - 1,
            atom_positions,
            atom_degrees,
            cell_size,
            output_partition,
            rcut,
            cuts_now,
            atom_indices=right_indices,
            origin=origin_right,
        )


class Node:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self, level=0):
        """String representation of the tree for visualization."""
        result = " " * (4 * level) + f"{self.value}\n"
        if self.left:
            result += self.left.__repr__(level + 1)
        if self.right:
            result += self.right.__repr__(level + 1)
        return result


def get_tree_constraint(level_values):
    """
    Builds a binary tree from a list of level values, where all nodes
    at the same level share the same value.

    Parameters
    ----------
    level_values : list
        List of values for each level of the tree

    Returns
    -------
    root : Node
        Root of the binary tree

    """
    if not level_values:
        return None

    root = Node(level_values[0])
    queue = deque([root])
    level = 1

    while queue and level < len(level_values):
        num_nodes_in_level = len(queue)  # Nodes in the current level
        value = level_values[level]

        for _ in range(num_nodes_in_level):
            parent = queue.popleft()

            # Create left and right child with the current level's value
            parent.left = Node(value)
            parent.right = Node(value)

            # Add children to the queue
            queue.append(parent.left)
            queue.append(parent.right)

        level += 1

    return root


def get_tree(array, level, index=0):
    """Recursively build a binary tree from an array.

    Parameters
    ----------
    array : list
        List of values for each node of the tree
    level : int
        Depth of the tree
    inedex : int, optional
        Index of the current node in the array

    Returns
    -------
    root : Node
        Root of the binary tree

    """
    if level == 0:
        return None
    root = Node(array[index])
    root.left = get_tree(array, level - 1, 2 * index + 1)
    root.right = get_tree(array, level - 1, 2 * index + 2)
    return root


def get_partition_from_tree(
    level,
    atom_indices,
    atom_positions,
    atom_degrees,
    decision_tree,
    cell_size,
    origin,
    output_partition,
):
    """Recursively partition the domain based on a decision tree.

    Parameters
    ----------
    level : int
        Depth of the tree
    atom_indices : NDArray
        Indices of the atoms
    atom_positions : NDArray
        Position of all the atoms
    atom_degrees : NDArray
        Atom degrees of all the atoms
    decision_tree : Node
        Root of the decision tree
    cell_size : NDArray
        Size of the cell
    origin : NDArray
        Origin of the cut
    output_partition : list
        Resulting output_partition

    """

    if level == 0:
        output_partition.append(atom_indices)
        return

    # test all possible cuts
    cut_position, left_indices, right_indices = get_cut_position(
        atom_positions,
        atom_degrees,
        atom_indices,
        decision_tree.value,
        cell_size,
        origin=origin,
    )

    origin_left = origin.copy()
    origin_right = origin.copy()
    origin_right[decision_tree.value] = cut_position

    get_partition_from_tree(
        level - 1,
        left_indices,
        atom_positions,
        atom_degrees,
        decision_tree.left,
        cell_size,
        origin_left,
        output_partition,
    )

    get_partition_from_tree(
        level - 1,
        right_indices,
        atom_positions,
        atom_degrees,
        decision_tree.right,
        cell_size,
        origin_right,
        output_partition,
    )


def partition_bruteforce(
    levels,
    atom_positions,
    atom_indices,
    adj_matrix,
    atom_degrees,
    cell_size,
    same_value_levels=True,
    origin=None,
    criterion="num_neighbors",
):
    """Partitions the domain based on a decision tree.

    Parameters
    ----------
    levels : int
        Depth of the tree
    atom_positions : NDArray
        Position of all the atoms
    atom_indices : NDArray
        Indices of the atoms
    adj_matrix : NDArray
        Adjacency matrix of the atomic structure
    atom_degrees : NDArray
        Atom degrees of all the atoms
    cell_size : NDArray
        Size of the cell
    same_value_levels : bool, optional
        If True, all nodes at the same level share the same value
    origin : NDArray, optional
        Origin of the cut
    criterion : str, optional
        Criterion to use for partitioning,
        either 'num_neighbors' or 'comm_volume'

    Returns
    -------
    best_partitition : list
        Best partition based on the minimum communication volume
    best_tree : Node
        Best decision tree
    best_comm : int
        Minimum communication volume

    """

    if origin is None:
        origin = np.array(
            [
                np.min(atom_positions[:, 0]),
                np.min(atom_positions[:, 1]),
                np.min(atom_positions[:, 2]),
            ]
        )

    if same_value_levels:
        # generate all possible decisions trees with the same value at each level
        level_combinations = product([0, 1, 2], repeat=levels)
        trees = [
            get_tree_constraint(level_values) for level_values in level_combinations
        ]

    else:
        # generate all possible decisions trees
        num_nodes = 2**levels - 1
        values = [0, 1, 2]
        all_combinations = np.array(np.meshgrid(*[values] * num_nodes)).T.reshape(
            -1, num_nodes
        )
        trees = [get_tree(combination, levels) for combination in all_combinations]

    # test possible paritions
    partitions = []
    comm_volumes = []

    for i, tree in enumerate(trees):
        print(f"Brute force iteration {i}/{len(trees)}")

        partition = []
        get_partition_from_tree(
            levels,
            atom_indices,
            atom_positions,
            atom_degrees,
            tree,
            cell_size,
            origin,
            partition,
        )
        partitions.append(partition)
        if criterion == "num_neighbors":
            comm_volumes.append(get_num_neighbors(adj_matrix, partition))
        elif criterion == "comm_volume":
            comm_volumes.append(get_comm_volume(adj_matrix, partition))
        else:
            raise ValueError(f"Criterion {criterion} not supported")

    # select the partition with the minimum communication volume
    best_partitition = partitions[np.argmin(comm_volumes)]
    best_tree = trees[np.argmin(comm_volumes)]
    best_comm = comm_volumes[np.argmin(comm_volumes)]

    return best_partitition, best_tree, best_comm


def partition_local_optimal(
    levels,
    atom_positions,
    atom_degrees,
    cell_size,
    adj_matrix,
    origin=None,
    criterion="num_neighbors",
):
    """Iteratively step through the decision tree to partition the domain

    Parameters
    ----------
    levels : int
        Depth of the tree
    atom_positions : NDArray
        Position of all the atoms
    atom_degrees : NDArray
        Atom degrees of all the atoms
    cell_size : NDArray
        Size of the cell
    adj_matrix : NDArray
        Adjacency matrix of the atomic structure
    origin : NDArray, optional
        Origin of the cut
    criterion : str, optional
        Criterion to use for partitioning,
        either 'num_neighbors' or 'comm_volume'

    Returns
    -------
    partition : list
        Best partition based on the minimum communication volume

    """
    partition = deque()
    partition.append(np.arange(atom_positions.shape[0]))

    origins = deque()
    if origin is None:
        origin = np.zeros(3)
    origins.append(origin)

    # step through the decision tree
    for _ in range(0, levels):

        # number of leaf nodes at this level
        num_partitions = len(partition)

        # loop over all partitions at this level
        # split from left to right and choose the best split
        for _ in range(num_partitions):
            # current partitition to devide
            domain_to_split = partition.popleft()
            origin = origins.popleft()

            temp_atom_pos = atom_positions.copy()
            for p in temp_atom_pos:
                for i in range(3):
                    if p[i] < origin[i]:
                        p[i] += cell_size[i]

            lx = np.max(temp_atom_pos[domain_to_split][:, 0]) - np.min(
                temp_atom_pos[domain_to_split][:, 0]
            )
            ly = np.max(temp_atom_pos[domain_to_split][:, 1]) - np.min(
                temp_atom_pos[domain_to_split][:, 1]
            )
            lz = np.max(temp_atom_pos[domain_to_split][:, 2]) - np.min(
                temp_atom_pos[domain_to_split][:, 2]
            )
            sub_domain_size = np.array([lx, ly, lz])

            # loop over cut possibilities
            new_partitions = [partition.copy() for _ in range(3)]
            new_origins = [origins.copy() for _ in range(3)]
            for k in range(3):
                cut_position, left_indices, right_indices = get_cut_position(
                    atom_positions,
                    atom_degrees,
                    domain_to_split,
                    k,
                    cell_size,
                    origin=origin,
                )
                origin_left = origin.copy()
                origin_right = origin.copy()
                origin_right[k] = cut_position
                new_partitions[k].append(left_indices)
                new_partitions[k].append(right_indices)
                new_origins[k].append(origin_left)
                new_origins[k].append(origin_right)

            # choose the best split
            if criterion == "num_neighbors":
                volumes = [
                    get_num_neighbors(adj_matrix, list(par)) for par in new_partitions
                ]
            elif criterion == "comm_volume":
                volumes = [
                    get_comm_volume(adj_matrix, list(par)) for par in new_partitions
                ]
            else:
                raise ValueError(f"Criterion {criterion} not supported")

            min_volume = np.min(volumes)
            min_indices = np.where(volumes == min_volume)[0]

            best_idx = min_indices[np.argmax(sub_domain_size[min_indices])]

            partition = new_partitions[best_idx]
            origins = new_origins[best_idx]

    return partition


def sparse_matrix_to_adjlist(adj_matrix):
    """Converts a sparse adjacency matrix to an adjacency list.

    Parameters
    ----------
    adj_matrix : NDArray
        Sparse adjacency matrix

    Returns
    -------
    adjacency_list : list
        Adjacency list representation of the graph

    """

    adj_matrix = adj_matrix.tocoo()

    n_nodes = adj_matrix.shape[0]
    adjacency_list = [[] for _ in range(n_nodes)]

    rows = adj_matrix.row
    cols = adj_matrix.col

    for i, j in zip(rows, cols):
        if i != j:  # Exclude self-loops
            adjacency_list[i].append(j)
            adjacency_list[j].append(i)

    # Remove duplicates and convert to set to ensure unique neighbors
    adjacency_list = [list(set(neighbors)) for neighbors in adjacency_list]

    return adjacency_list


def partition_metis(
    num_partitions,
    adj_matrix,
):
    """Partitions the domain using the METIS library.

    Parameters
    ----------
    levels : int
        Depth of the tree
    adj_matrix : NDArray
        Adjacency matrix of the atomic structure

    Returns
    -------
    partition : list
        Best partition based on the METIS library
    """

    G = sparse_matrix_to_adjlist(adj_matrix)
    (_, parts) = pymetis.part_graph(num_partitions, adjacency=G)
    parts = np.array(parts)
    partition = [np.argwhere(parts == i).flatten() for i in range(num_partitions)]
    return partition


def one_dim_cut(
    levels,
    atom_positions,
    atom_degrees,
    cell_size,
    output_partition,
    cut_dim,
    atom_indices=None,
):
    """Partitions the domain based on the longest dimension.

    Parameters
    ----------
    levels : int
        number of levels to partition
    atom_positions : NDArray
        position of all the atoms
    atom_degrees : NDArray
        atom_degrees of all the atoms
    cell_size : NDArray
        size of the cell
    output_partition : list
        Resulting output_partition
    cut_dim : int
        dimension to cut
    atom_indices : NDArray, optional
        indices of the atoms

    """

    if atom_indices is None:
        atom_indices = np.arange(len(atom_positions))

    if levels == 0:
        output_partition.append(atom_indices)
        return

    _, left_indices, right_indices = get_cut_position(
        atom_positions, atom_degrees, atom_indices, cut_dim, cell_size
    )

    one_dim_cut(
        levels - 1,
        atom_positions,
        atom_degrees,
        cell_size,
        output_partition,
        cut_dim,
        atom_indices=left_indices,
    )
    one_dim_cut(
        levels - 1,
        atom_positions,
        atom_degrees,
        cell_size,
        output_partition,
        cut_dim,
        atom_indices=right_indices,
    )


def partition_longest_dim(
    levels,
    atom_positions,
    atom_degrees,
    cell_size,
    output_partition,
    atom_indices=None,
):
    """Partitions the domain based on the longest dimension.

    Parameters
    ----------
    levels : int
        number of levels to partition
    atom_positions : NDArray
        position of all the atoms
    atom_degrees : NDArray
        atom_degrees of all the atoms
    cell_size : NDArray
        size of the cell
    output_partition : list
        Resulting output_partition
    atom_indices : NDArray, optional
        indices of the atoms

    """

    lx = np.max(atom_positions[:, 0]) - np.min(atom_positions[:, 0])
    ly = np.max(atom_positions[:, 1]) - np.min(atom_positions[:, 1])
    lz = np.max(atom_positions[:, 2]) - np.min(atom_positions[:, 2])
    domain_size = np.array([lx, ly, lz])

    dim = np.argmax(domain_size)

    one_dim_cut(
        levels,
        atom_positions,
        atom_degrees,
        cell_size,
        output_partition,
        dim,
        atom_indices=atom_indices,
    )


def parition_wrapper(
    levels: int,
    atom_positions: np.ndarray,
    cell_size: np.ndarray,
    adj_matrix: np.ndarray,
    rcut: float,
    method: str,
    criterion: str,
):
    """Wrapper function to partition the domain based on the method.

    Parameters
    ----------
    levels : int
        Depth of the tree
    atom_positions : NDArray
        Position of all the atoms
    cell_size : NDArray
        Size of the cell
    adj_matrix : NDArray
        Adjacency matrix of the atomic structure
    rcut : float
        Cutoff radius
    method : str
        Partitioning method to use.
        Either 'local_optimal', 'bruteforce',
            'surface_volume', 'approx_neighbors',
            'old_neighbors', 'metis',
            or 'longest_dim'.
    
    criterion : str
        Criterion to use for partitioning,
        either 'num_neighbors' or 'comm_volume'.
        It only applies to the 'local_optimal' and 'bruteforce' methods.

    Returns
    -------
    partition : list
        Best partition based on the method

    Methods
    -------
    'old_neighbors': copy of previous custom which was on the _nccl branch, divides based on approximate neighbors in each dimention (2*rcut/width)
    'surface_volume': minimizes comm volume recursively. approximates it through the surface volume of the local cuboid.
                    surfaces results in rcut*surfaceDim, edges result in parts of parts of the cylinder with rcut radius
                    corners result in parts of the spheres with rcut radius
                    cost only incurred if not periodic in that dimension
    'approx_neighbors': a mixture between surface and 'old neighbors'. guesses neighbors by repeating own domain as surface. 
                        surfaces cost: rcut/dim_normal
                        edges: rcut/diagonal
                        similar for corners
                        cost only incurred if not periodic
    'local_optimal': not recursive, goes down decision tree layer by layer.
                    takes for each decision the local optimal one.
                    can choose either to minimize comm volume or neighbors.
                    computes it from adj. matrix.
    'bruteforce': test either all trees or trees with the same decision per level
                can choose either to minimize comm volume or neighbors
                computes it from adj. matrix
    """

    is_periodic = [True, True, True]
    cuts = [0, 0, 0]
    origin = np.array(
        [
            np.min(atom_positions[:, 0]),
            np.min(atom_positions[:, 1]),
            np.min(atom_positions[:, 2]),
        ]
    )
    atom_indices = np.arange(atom_positions.shape[0])
    atom_degrees = adj_matrix.tocsr().getnnz(axis=1)

    if method == "local_optimal":
        partition = partition_local_optimal(
            levels,
            atom_positions,
            atom_degrees,
            cell_size,
            adj_matrix,
            origin=origin,
            criterion=criterion,
        )
    elif method == "bruteforce":
        partition, _, _ = partition_bruteforce(
            levels,
            atom_positions,
            atom_indices,
            adj_matrix,
            atom_degrees,
            cell_size,
            origin=origin,
            criterion=criterion,
        )
    elif method == "surface_volume":
        partition = []
        partition_surface_volume(
            levels,
            atom_positions,
            atom_degrees,
            cell_size,
            partition,
            rcut,
            is_periodic,
            atom_indices,
        )
    elif method == "approx_neighbors":
        partition = []
        partition_approx_neighbors(
            levels,
            atom_positions,
            atom_degrees,
            cell_size,
            partition,
            rcut,
            is_periodic,
            atom_indices,
        )
    elif method == "old_neighbors":
        partition = []
        partition_old_neighbors(
            levels,
            atom_positions,
            atom_degrees,
            cell_size,
            partition,
            rcut,
            cuts,
            atom_indices,
            origin,
        )
    elif method == "longest_dim":
        partition = []
        partition_longest_dim(
            levels,
            atom_positions,
            atom_degrees,
            cell_size,
            partition,
            atom_indices,
        )
    elif method == "metis":
        partition = partition_metis(levels, adj_matrix) # levels = num_partitions here

    else:
        raise ValueError(f"Method {method} not supported")

    return partition