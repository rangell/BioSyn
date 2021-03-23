#cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np
from tqdm import tqdm
from IPython import embed


INT = np.int
BOOL = np.bool

ctypedef np.int_t INT_t
ctypedef np.npy_bool BOOL_t


@cython.boundscheck(False)
@cython.wraparound(False)
def _build_adj_index(np.ndarray[INT_t, ndim=1] values,
                              INT_t max_value):
    # requires: values in ascending order
    cdef INT_t index_size = max_value + 1
    cdef np.ndarray[INT_t, ndim=2] adj_index = np.zeros([index_size, 2], dtype=INT)
    cdef INT_t i = 0, c

    cdef INT_t curr_col = values[0]
    for i, c in enumerate(values):
        if c != curr_col:
            curr_col = c
            adj_index[curr_col, 0] = i
            adj_index[curr_col, 1] = i+1
        else:
            adj_index[curr_col, 1] += 1

    return adj_index

@cython.wraparound(False)
@cython.boundscheck(False)
def _has_entity_in_component(list stack,
                             np.ndarray[INT_t, ndim=1] to_vertices,
                             np.ndarray[INT_t, ndim=2] adj_index,
                             INT_t num_entities,
                             INT_t original_to_node):
    # performs DFS and returns `True` whenever it hits an entity
    cdef set visited = set()
    cdef bint found = False
    cdef INT_t curr_node
    cdef INT_t original_node = stack[len(stack) - 1]
    while len(stack) > 0:
        # pop
        curr_node = stack[len(stack) - 1]
        stack = stack[:len(stack) - 1]

        # check if `curr_node` is an entity
        if curr_node < num_entities:
            found = True
            # if original_to_node < num_entities:
                # print(f"Found: {original_node} can reach {curr_node}. Dropping {original_to_node}")
            break

        # check if we've visited `curr_node`
        if curr_node in visited:
            continue
        visited.add(curr_node)

        # get neighbors of `curr_node` and push them onto the stack
        start_idx, end_idx = adj_index[curr_node, 0], adj_index[curr_node, 1]
        stack.extend(to_vertices[start_idx:end_idx].tolist())

    return found


@cython.boundscheck(False)
@cython.wraparound(False)
def special_partition(np.ndarray[INT_t, ndim=1] row, 
                      np.ndarray[INT_t, ndim=1] col,
                      np.ndarray[INT_t, ndim=1] ordered_indices,
                      INT_t num_entities):
    assert row.shape[0] == col.shape[0]
    assert row.shape[0] == ordered_indices.shape[0]

    cdef INT_t num_edges = row.shape[0]
    cdef np.ndarray[BOOL_t, ndim=1] keep_mask = np.ones([num_edges,], dtype=BOOL)
    cdef np.ndarray[INT_t, ndim=1] tmp_col
    cdef INT_t r, c
    cdef bint entity_reachable, can_now_reach
    cdef INT_t row_max_value = row[-1]

    # has shape [N, 2]; [:,0] are starting indices and [:,1] are (exclusive) ending indices
    cdef np.ndarray[INT_t, ndim=2] row_wise_adj_index
    row_wise_adj_index = _build_adj_index(
            row, row_max_value
    )

    for i in tqdm(ordered_indices, desc='Paritioning Joint Graph'):
        r = row[i]
        c = col[i]

        assert r >= num_entities

        # Check if entity is reachable to begin with
        entity_reachable = _has_entity_in_component(
            [r], col[keep_mask], row_wise_adj_index, num_entities, c)
        assert entity_reachable

        
        # try removing both the forward and backward edges
        keep_mask[i] = False

        # update the adj list index for the forward and backward edges
        # col_wise_adj_index[c:, :] -= 1
        # col_wise_adj_index[c, 0] += 1
        row_wise_adj_index[r:, :] -= 1
        row_wise_adj_index[r, 0] += 1

        # create the temporary graph we want to check
        # tmp_row = row[keep_mask]
        tmp_col = col[keep_mask]

        # check if we can remove the edge (r, c) 
        # if c < num_entities:
            # print(f"Checking if {r},{c} can be dropped")
        entity_reachable = _has_entity_in_component(
            [r], tmp_col, row_wise_adj_index, num_entities, c)

        # add the edge back if we need it
        if not entity_reachable:
            # print("Not found")
            keep_mask[i] = True
            row_wise_adj_index[r:, :] += 1
            row_wise_adj_index[r, 0] -= 1
            # Check if entity is now reachable
            can_now_reach = _has_entity_in_component(
                [r], col[keep_mask], row_wise_adj_index, num_entities, c)
            assert can_now_reach

    return keep_mask
