import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

max_elements = 83

def single_matrix2label(
    orbital_template,
    fock_block_offsets, 
    idx_to_atomic_number,
    fock_block_rows, 
    fock_block_cols,
    fock_matrix,
    target,
    forward
):

    # iterates over atom pairs (i, j) within this molecule
    for idx in range(len(fock_block_rows)):
        i = fock_block_rows[idx]
        j = fock_block_cols[idx]

        # extract this atom pair subblock from the fock matrix:
        row_slice = slice(
            fock_block_offsets[i], fock_block_offsets[i + 1]
        )
        col_slice = slice(
            fock_block_offsets[j], fock_block_offsets[j + 1]
        )
        interaction_block = fock_matrix[row_slice, col_slice]

        # Get the template of all orbital interactions for this atom pair
        atomic_element_i = idx_to_atomic_number[i]
        atomic_element_j = idx_to_atomic_number[j]

        element_interaction_key = atomic_element_j + max_elements * atomic_element_i

        for row_slice_block, col_slice_block, output_slice_block in orbital_template[element_interaction_key]:

                if forward == True:
                    # single orbital interaction (e.g. s-p, d-d, etc.)
                    block = interaction_block[
                        row_slice_block,
                        col_slice_block
                    ]

                    target[idx][
                        output_slice_block
                    ] = block.flatten()
                else:
                    # single orbital interaction (e.g. s-p, d-d, etc.)
                    interaction_block[
                        row_slice_block,
                        col_slice_block
                    ] = target[idx][
                        output_slice_block
                    ].reshape(
                        row_slice_block.stop - row_slice_block.start,
                        col_slice_block.stop - col_slice_block.start
                    )


def multiple_matrix2label(
    num_structures,
    orbital_template,
    fock_block_offsets, 
    idx_to_atomic_number,
    fock_block_rows, 
    fock_block_cols,
    fock_matrix,
    target,
    forward
):
    for k in range(num_structures):

        # iterates over atom pairs (i, j) within this molecule
        for idx in range(len(fock_block_rows[k])):
            i = fock_block_rows[k][idx]
            j = fock_block_cols[k][idx]

            # extract this atom pair subblock from the fock matrix:
            row_slice = slice(
                fock_block_offsets[k][i], fock_block_offsets[k][i + 1]
            )
            col_slice = slice(
                fock_block_offsets[k][j], fock_block_offsets[k][j + 1]
            )
            interaction_block = fock_matrix[k][row_slice, col_slice]

            # Get the template of all orbital interactions for this atom pair
            atomic_element_i = idx_to_atomic_number[i]
            atomic_element_j = idx_to_atomic_number[j]

            element_interaction_key = atomic_element_j + max_elements * atomic_element_i

            for row_slice_block, col_slice_block, output_slice_block in orbital_template[element_interaction_key]:

                    if forward == True:
                        # single orbital interaction (e.g. s-p, d-d, etc.)
                        block = interaction_block[
                            row_slice_block,
                            col_slice_block
                        ]

                        target[k][idx][
                            output_slice_block
                        ] = block.flatten()
                    else:
                        # single orbital interaction (e.g. s-p, d-d, etc.)
                        interaction_block[
                            row_slice_block,
                            col_slice_block
                        ] = target[k][idx][
                            output_slice_block
                        ].reshape(
                            row_slice_block.stop - row_slice_block.start,
                            col_slice_block.stop - col_slice_block.start
                        )
