"""MPI-distributed 3D structure-function grid generation.

Phase 1 uses separation distribution:
- every rank holds the full velocity field
- ranks own subsets of ``(lx, ly)`` pairs
- each rank loops over all ``lz``
- rank 0 gathers packed local blocks and assembles the full SF grid
"""

from __future__ import annotations

import numpy as np

from .reducers_3d import (
    compute_velocity_sf_reduction_3d,
    finalize_velocity_sf_reduction,
)
from .separation_map import compute_separation_pairs_for_rank, validate_processor_grid


def _require_mpi():
    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - exercised in environments without mpi4py
        raise ImportError(
            "mpi4py is required for the MPI backend. Install with `python -m pip install mpi4py`."
        ) from exc
    return MPI

def _compute_velocity_structure_functions(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    shift_x: int,
    shift_y: int,
    shift_z: int,
    sf_type: tuple[str, ...],
    boundary=None,
) -> dict[str, float]:
    """Compute velocity structure-function values for one displacement vector."""
    reductions = compute_velocity_sf_reduction_3d(
        u,
        v,
        w,
        shift_x,
        shift_y,
        shift_z,
        sf_type,
        boundary=boundary,
    )
    return finalize_velocity_sf_reduction(reductions, sf_type)


def generate_sf_grid_3d_mpi(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    sf_type: list[str] | tuple[str, ...] = ("LL",),
    px: int = 1,
    boundary=None,
    comm=None,
) -> dict[str, np.ndarray]:
    """Generate 3D velocity structure-function grids using MPI separation distribution.

    Parameters
    ----------
    u, v, w:
        Full 3D velocity arrays with common shape ``(Nx, Ny, Nz)``.
    x, y, z:
        1D coordinates for the three directions.
    sf_type:
        Velocity structure-function types to compute. Supported values are
        ``LL``, ``TT``, ``LLL``, and ``LTT``.
    px:
        Number of processor columns in the x direction of the separation grid.
    boundary:
        Optional boundary mode. ``None`` uses non-periodic sliced differences.
        ``"periodic-all"`` uses wrapped differences to match the legacy serial
        API behavior.
    comm:
        Optional MPI communicator. Defaults to ``MPI.COMM_WORLD``.

    Returns
    -------
    dict
        On rank 0, a dictionary containing ``x-diffs``, ``y-diffs``, ``z-diffs``
        and one 3D array per requested structure-function type. On non-root ranks,
        all values are ``None``.
    """
    if comm is None:
        _require_mpi()
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    sf_type = tuple(sf_type)
    supported = {"LL", "TT", "LLL", "LTT"}
    unsupported = set(sf_type) - supported
    if unsupported:
        raise ValueError(
            "Phase 1 MPI backend currently supports only velocity SF types: "
            "LL, TT, LLL, LTT."
        )

    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("u, v, and w must have identical shapes.")
    if u.ndim != 3:
        raise ValueError("u, v, and w must be 3D arrays.")

    nx_half, ny_half = validate_processor_grid(len(x), len(y), px, nprocs)
    nz_half = len(z) // 2
    if nz_half == 0:
        raise ValueError("z must contain at least two points.")

    local_pairs = compute_separation_pairs_for_rank(len(x), len(y), px, nprocs, rank)
    local_nxy = local_pairs.shape[0]
    n_orders = len(sf_type)
    local_values = np.empty((local_nxy, nz_half, n_orders), dtype=np.float64)

    for pair_id, (shift_x, shift_y) in enumerate(local_pairs):
        for shift_z in range(nz_half):
            if shift_x == 0 and shift_y == 0 and shift_z == 0:
                local_values[pair_id, shift_z, :] = 0.0
                continue
            sf_dict = _compute_velocity_structure_functions(
                u,
                v,
                w,
                int(shift_x),
                int(shift_y),
                int(shift_z),
                sf_type,
                boundary=boundary,
            )
            local_values[pair_id, shift_z, :] = [sf_dict[f"SF_{name}"] for name in sf_type]

    gathered_pairs = comm.gather(local_pairs, root=0)
    gathered_values = comm.gather(local_values, root=0)

    if rank != 0:
        return {
            "x-diffs": None,
            "y-diffs": None,
            "z-diffs": None,
            **{f"SF_{name}_grid": None for name in sf_type},
        }

    output = {
        "x-diffs": np.asarray(x[:nx_half], dtype=np.float64) - float(x[0]),
        "y-diffs": np.asarray(y[:ny_half], dtype=np.float64) - float(y[0]),
        "z-diffs": np.asarray(z[:nz_half], dtype=np.float64) - float(z[0]),
    }
    for name in sf_type:
        output[f"SF_{name}_grid"] = np.zeros((nx_half, ny_half, nz_half), dtype=np.float64)

    for rank_pairs, rank_values in zip(gathered_pairs, gathered_values):
        for pair_id, (shift_x, shift_y) in enumerate(rank_pairs):
            for order_id, name in enumerate(sf_type):
                output[f"SF_{name}_grid"][shift_x, shift_y, :] = rank_values[pair_id, :, order_id]

    return output
