"""Utilities for distributing separation pairs across MPI ranks.

This module ports the separation-pair scheduling idea used by ``fastSF`` into a
small, testable Python API. The mapping is intentionally independent of
``mpi4py`` so it can be unit tested without an MPI runtime.
"""

from __future__ import annotations

import numpy as np


def validate_processor_grid(nx: int, ny: int, px: int, nprocs: int) -> tuple[int, int]:
    """Validate the processor-grid inputs used for separation assignment.

    Parameters
    ----------
    nx, ny:
        Full grid sizes in the two displacement directions.
    px:
        Number of processor columns in the x direction.
    nprocs:
        Total number of MPI ranks.

    Returns
    -------
    tuple[int, int]
        The half-grid sizes ``(nx_half, ny_half)`` used by the displacement map.
    """
    if px <= 0:
        raise ValueError("px must be a positive integer.")
    if nprocs <= 0:
        raise ValueError("nprocs must be a positive integer.")
    if nprocs % px != 0:
        raise ValueError("px must divide the total number of processors.")

    py = nprocs // px
    nx_half = nx // 2
    ny_half = ny // 2

    if nx_half == 0 or ny_half == 0:
        raise ValueError("Grid must have at least two points in each mapped direction.")
    if nx_half % px != 0:
        raise ValueError("nx//2 must be divisible by px.")
    if ny_half % py != 0:
        raise ValueError("ny//2 must be divisible by py.")

    return nx_half, ny_half


def compute_rank_coordinates(rank: int, py: int) -> tuple[int, int]:
    """Return the processor-grid coordinates for a linear rank id."""
    if py <= 0:
        raise ValueError("py must be a positive integer.")
    if rank < 0:
        raise ValueError("rank must be non-negative.")

    rank_y = rank % py
    rank_x = (rank - rank_y) // py
    return rank_x, rank_y


def compute_axis_index_list(n_half: int, nproc_axis: int, rank_axis: int) -> np.ndarray:
    """Return the displacement indices assigned to one processor-grid coordinate.

    The ordering mirrors ``fastSF``:
    - take every ``nproc_axis``-th index starting from ``rank_axis``
    - interleave the corresponding mirrored indices from the upper end

    Examples
    --------
    ``n_half=8, nproc_axis=2`` yields:
    - ``rank_axis=0`` -> ``[0, 7, 4, 3]``
    - ``rank_axis=1`` -> ``[1, 6, 5, 2]``
    """
    if n_half <= 0:
        raise ValueError("n_half must be positive.")
    if nproc_axis <= 0:
        raise ValueError("nproc_axis must be positive.")
    if rank_axis < 0 or rank_axis >= nproc_axis:
        raise ValueError("rank_axis must be in [0, nproc_axis).")
    if n_half % nproc_axis != 0:
        raise ValueError("n_half must be divisible by nproc_axis.")

    list_size = n_half // nproc_axis
    index_list = np.empty((list_size,), dtype=np.int64)

    if nproc_axis == n_half:
        index_list[0] = rank_axis
        return index_list

    for i in range(0, list_size, 2):
        base_index = rank_axis + i * nproc_axis
        index_list[i] = base_index
        if i + 1 < list_size:
            index_list[i + 1] = n_half - 1 - base_index

    return index_list


def compute_separation_pairs_for_rank(
    nx: int,
    ny: int,
    px: int,
    nprocs: int,
    rank: int,
) -> np.ndarray:
    """Return the ``(lx, ly)`` separation pairs assigned to one rank."""
    nx_half, ny_half = validate_processor_grid(nx, ny, px, nprocs)

    if rank < 0 or rank >= nprocs:
        raise ValueError("rank must be in [0, nprocs).")

    py = nprocs // px
    rank_x, rank_y = compute_rank_coordinates(rank, py)
    x_indices = compute_axis_index_list(nx_half, px, rank_x)
    y_indices = compute_axis_index_list(ny_half, py, rank_y)

    pairs = np.empty((x_indices.size * y_indices.size, 2), dtype=np.int64)
    cursor = 0
    for x_index in x_indices:
        next_cursor = cursor + y_indices.size
        pairs[cursor:next_cursor, 0] = x_index
        pairs[cursor:next_cursor, 1] = y_indices
        cursor = next_cursor

    return pairs


def compute_separation_map(nx: int, ny: int, px: int, nprocs: int) -> list[np.ndarray]:
    """Return the full rank-to-separation-pairs assignment map."""
    validate_processor_grid(nx, ny, px, nprocs)
    return [
        compute_separation_pairs_for_rank(nx, ny, px, nprocs, rank)
        for rank in range(nprocs)
    ]
