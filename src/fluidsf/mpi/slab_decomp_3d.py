"""Z-slab decomposition helpers for 3D MPI structure-function reductions."""

from __future__ import annotations

import numpy as np

from .reducers_3d import finalize_structure_function_reduction


def compute_slab_bounds_1d(global_size: int, nprocs: int, rank: int) -> tuple[int, int]:
    """Return the half-open slab bounds for a 1D block decomposition."""
    if nprocs <= 0:
        raise ValueError("nprocs must be positive.")
    if rank < 0 or rank >= nprocs:
        raise ValueError("rank must satisfy 0 <= rank < nprocs.")

    base = global_size // nprocs
    remainder = global_size % nprocs
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)
    return start, stop


def extract_local_z_slab(arr: np.ndarray, nprocs: int, rank: int) -> np.ndarray:
    """Return the local z-slab owned by one rank."""
    start, stop = compute_slab_bounds_1d(arr.shape[2], nprocs, rank)
    return np.ascontiguousarray(arr[:, :, start:stop])


def _propagate_periodic_halo_z(
    local_arr: np.ndarray, halo_depth: int, comm, direction: str
) -> np.ndarray:
    """Collect periodic halo cells from successive ranks in one z direction."""
    if halo_depth < 0:
        raise ValueError("halo_depth must be non-negative.")
    if halo_depth == 0:
        return np.empty(local_arr.shape[:2] + (0,), dtype=local_arr.dtype)

    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        if direction == "forward":
            return np.ascontiguousarray(local_arr[:, :, :halo_depth])
        return np.ascontiguousarray(local_arr[:, :, -halo_depth:])

    if direction == "forward":
        dest = (rank - 1) % size
        source = (rank + 1) % size
    elif direction == "backward":
        dest = (rank + 1) % size
        source = (rank - 1) % size
    else:
        raise ValueError("direction must be 'forward' or 'backward'.")

    pieces = []
    remaining = halo_depth
    propagated = np.ascontiguousarray(local_arr)
    hop = 0

    while remaining > 0:
        recvbuf = np.empty_like(propagated)
        comm.Sendrecv(
            sendbuf=propagated,
            dest=dest,
            sendtag=hop,
            recvbuf=recvbuf,
            source=source,
            recvtag=hop,
        )
        take = min(remaining, recvbuf.shape[2])
        if direction == "forward":
            pieces.append(np.ascontiguousarray(recvbuf[:, :, :take]))
        else:
            pieces.append(np.ascontiguousarray(recvbuf[:, :, -take:]))
        remaining -= take
        propagated = recvbuf
        hop += 1

    if direction == "forward":
        return np.concatenate(pieces, axis=2)
    pieces.reverse()
    return np.concatenate(pieces, axis=2)


def exchange_periodic_halo_z(local_arr: np.ndarray, halo_depth: int, comm) -> np.ndarray:
    """Extend a local z-slab with periodic halo cells on both sides."""
    left_halo = _propagate_periodic_halo_z(local_arr, halo_depth, comm, "backward")
    right_halo = _propagate_periodic_halo_z(local_arr, halo_depth, comm, "forward")
    return np.concatenate((left_halo, local_arr, right_halo), axis=2)


def _sum_and_count(values: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(values)
    return float(np.sum(values[valid], dtype=np.float64)), int(np.count_nonzero(valid))


def _require_mpi():
    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - exercised where mpi4py is missing
        raise ImportError(
            "mpi4py is required for the slab-decomposition MPI backend."
        ) from exc
    return MPI


def compute_velocity_sf_reduction_3d_periodic_z_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    w_local: np.ndarray,
    shift_x: int,
    shift_y: int,
    shift_z: int,
    sf_type: tuple[str, ...],
    *,
    comm=None,
) -> dict[str, float]:
    """Compute one periodic 3D velocity SF using z-slab decomposition and MPI."""
    if comm is None:
        _require_mpi()
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    if shift_x == 0 and shift_y == 0 and shift_z == 0:
        raise ValueError("The zero displacement vector is not valid for SF evaluation.")

    if u_local.shape != v_local.shape or u_local.shape != w_local.shape:
        raise ValueError("u_local, v_local, and w_local must have identical shapes.")
    if u_local.ndim != 3:
        raise ValueError("u_local, v_local, and w_local must be 3D arrays.")

    halo_depth = int(shift_z)
    u_ext = exchange_periodic_halo_z(u_local, halo_depth, comm)
    v_ext = exchange_periodic_halo_z(v_local, halo_depth, comm)
    w_ext = exchange_periodic_halo_z(w_local, halo_depth, comm)

    core_start = halo_depth
    core_stop = core_start + u_local.shape[2]
    base_u = u_ext[:, :, core_start:core_stop]
    base_v = v_ext[:, :, core_start:core_stop]
    base_w = w_ext[:, :, core_start:core_stop]

    target_u = u_ext[:, :, core_start + shift_z : core_stop + shift_z]
    target_v = v_ext[:, :, core_start + shift_z : core_stop + shift_z]
    target_w = w_ext[:, :, core_start + shift_z : core_stop + shift_z]

    du = np.roll(target_u, shift=(-shift_x, -shift_y), axis=(0, 1)) - base_u
    dv = np.roll(target_v, shift=(-shift_x, -shift_y), axis=(0, 1)) - base_v
    dw = np.roll(target_w, shift=(-shift_x, -shift_y), axis=(0, 1)) - base_w

    lx = float(shift_x)
    ly = float(shift_y)
    lz = float(shift_z)
    radius = np.sqrt(lx * lx + ly * ly + lz * lz)

    d_long = (lx * du + ly * dv + lz * dw) / radius
    d_perp_sq = du * du + dv * dv + dw * dw - d_long * d_long
    d_perp_sq = np.maximum(d_perp_sq, 0.0)

    local_reductions: dict[str, float | int] = {}
    if "LL" in sf_type:
        local_reductions["SF_LL_sum"], local_reductions["SF_LL_count"] = _sum_and_count(
            d_long**2
        )
    if "TT" in sf_type:
        local_reductions["SF_TT_sum"], local_reductions["SF_TT_count"] = _sum_and_count(
            d_perp_sq
        )
    if "LLL" in sf_type:
        local_reductions["SF_LLL_sum"], local_reductions["SF_LLL_count"] = _sum_and_count(
            d_long**3
        )
    if "LTT" in sf_type:
        local_reductions["SF_LTT_sum"], local_reductions["SF_LTT_count"] = _sum_and_count(
            d_long * d_perp_sq
        )

    global_reductions = {
        key: comm.allreduce(value) for key, value in local_reductions.items()
    }
    return finalize_structure_function_reduction(global_reductions, sf_type)


def compute_scalar_sf_reduction_3d_periodic_z_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    w_local: np.ndarray,
    scalar_local: np.ndarray,
    shift_x: int,
    shift_y: int,
    shift_z: int,
    sf_type: tuple[str, ...],
    *,
    comm=None,
) -> dict[str, float]:
    """Compute one periodic 3D scalar SF using z-slab decomposition and MPI."""
    if comm is None:
        _require_mpi()
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    if shift_x == 0 and shift_y == 0 and shift_z == 0:
        raise ValueError("The zero displacement vector is not valid for SF evaluation.")

    if (
        u_local.shape != v_local.shape
        or u_local.shape != w_local.shape
        or u_local.shape != scalar_local.shape
    ):
        raise ValueError("u_local, v_local, w_local, and scalar_local must match.")
    if u_local.ndim != 3:
        raise ValueError("u_local, v_local, w_local, and scalar_local must be 3D arrays.")

    halo_depth = int(shift_z)
    u_ext = exchange_periodic_halo_z(u_local, halo_depth, comm)
    v_ext = exchange_periodic_halo_z(v_local, halo_depth, comm)
    w_ext = exchange_periodic_halo_z(w_local, halo_depth, comm)
    s_ext = exchange_periodic_halo_z(scalar_local, halo_depth, comm)

    core_start = halo_depth
    core_stop = core_start + u_local.shape[2]
    base_u = u_ext[:, :, core_start:core_stop]
    base_v = v_ext[:, :, core_start:core_stop]
    base_w = w_ext[:, :, core_start:core_stop]
    base_s = s_ext[:, :, core_start:core_stop]

    target_u = u_ext[:, :, core_start + shift_z : core_stop + shift_z]
    target_v = v_ext[:, :, core_start + shift_z : core_stop + shift_z]
    target_w = w_ext[:, :, core_start + shift_z : core_stop + shift_z]
    target_s = s_ext[:, :, core_start + shift_z : core_stop + shift_z]

    du = np.roll(target_u, shift=(-shift_x, -shift_y), axis=(0, 1)) - base_u
    dv = np.roll(target_v, shift=(-shift_x, -shift_y), axis=(0, 1)) - base_v
    dw = np.roll(target_w, shift=(-shift_x, -shift_y), axis=(0, 1)) - base_w
    ds = np.roll(target_s, shift=(-shift_x, -shift_y), axis=(0, 1)) - base_s

    lx = float(shift_x)
    ly = float(shift_y)
    lz = float(shift_z)
    radius = np.sqrt(lx * lx + ly * ly + lz * lz)

    d_long = (lx * du + ly * dv + lz * dw) / radius
    local_reductions: dict[str, float | int] = {}
    if "SS" in sf_type:
        local_reductions["SF_SS_sum"], local_reductions["SF_SS_count"] = _sum_and_count(
            ds**2
        )
    if "LSS" in sf_type:
        local_reductions["SF_LSS_sum"], local_reductions["SF_LSS_count"] = _sum_and_count(
            d_long * ds**2
        )

    global_reductions = {
        key: comm.allreduce(value) for key, value in local_reductions.items()
    }
    return finalize_structure_function_reduction(global_reductions, sf_type)


def generate_sf_grid_3d_periodic_z_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    w_local: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    scalar_local: np.ndarray | None = None,
    sf_type: list[str] | tuple[str, ...] = ("LL",),
    comm=None,
) -> dict[str, np.ndarray]:
    """Generate a periodic 3D velocity SF grid from local z-slabs."""
    if comm is None:
        _require_mpi()
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    if u_local.shape != v_local.shape or u_local.shape != w_local.shape:
        raise ValueError("u_local, v_local, and w_local must have identical shapes.")
    if u_local.ndim != 3:
        raise ValueError("u_local, v_local, and w_local must be 3D arrays.")

    sf_type = tuple(sf_type)
    supported = {"LL", "TT", "LLL", "LTT", "SS", "LSS"}
    unsupported = set(sf_type) - supported
    if unsupported:
        raise ValueError(
            "The z-slab MPI grid backend currently supports only velocity SF "
            "types: LL, TT, LLL, LTT, SS, LSS."
        )

    rank = comm.Get_rank()
    global_nz = comm.allreduce(u_local.shape[2])
    if global_nz != len(z):
        raise ValueError("The local z-slab sizes do not sum to len(z).")
    if u_local.shape[0] != len(x) or u_local.shape[1] != len(y):
        raise ValueError("Local slab x/y dimensions must match len(x) and len(y).")
    if scalar_local is not None and scalar_local.shape != u_local.shape:
        raise ValueError("scalar_local must match the local velocity slab shape.")
    if any(name in sf_type for name in ("SS", "LSS")) and scalar_local is None:
        raise ValueError("scalar_local is required when requesting SS or LSS.")

    nx_half = len(x) // 2
    ny_half = len(y) // 2
    nz_half = len(z) // 2
    if nx_half == 0 or ny_half == 0 or nz_half == 0:
        raise ValueError("x, y, and z must each contain at least two points.")

    output = {
        "x-diffs": np.asarray(x[:nx_half], dtype=np.float64) - float(x[0]),
        "y-diffs": np.asarray(y[:ny_half], dtype=np.float64) - float(y[0]),
        "z-diffs": np.asarray(z[:nz_half], dtype=np.float64) - float(z[0]),
    }
    for name in sf_type:
        output[f"SF_{name}_grid"] = np.zeros((nx_half, ny_half, nz_half), dtype=np.float64)

    for shift_x in range(nx_half):
        for shift_y in range(ny_half):
            for shift_z in range(nz_half):
                if shift_x == 0 and shift_y == 0 and shift_z == 0:
                    continue
                reduced = {}
                velocity_sf_type = tuple(
                    name for name in sf_type if name in ("LL", "TT", "LLL", "LTT")
                )
                scalar_sf_type = tuple(name for name in sf_type if name in ("SS", "LSS"))
                if velocity_sf_type:
                    reduced.update(
                        compute_velocity_sf_reduction_3d_periodic_z_slab_mpi(
                            u_local,
                            v_local,
                            w_local,
                            shift_x,
                            shift_y,
                            shift_z,
                            velocity_sf_type,
                            comm=comm,
                        )
                    )
                if scalar_sf_type:
                    reduced.update(
                        compute_scalar_sf_reduction_3d_periodic_z_slab_mpi(
                            u_local,
                            v_local,
                            w_local,
                            scalar_local,
                            shift_x,
                            shift_y,
                            shift_z,
                            scalar_sf_type,
                            comm=comm,
                        )
                    )
                if rank == 0:
                    for name in sf_type:
                        output[f"SF_{name}_grid"][shift_x, shift_y, shift_z] = reduced[
                            f"SF_{name}"
                        ]

    if rank != 0:
        return {
            "x-diffs": None,
            "y-diffs": None,
            "z-diffs": None,
            **{f"SF_{name}_grid": None for name in sf_type},
        }
    return output


__all__ = (
    "compute_slab_bounds_1d",
    "compute_scalar_sf_reduction_3d_periodic_z_slab_mpi",
    "generate_sf_grid_3d_periodic_z_slab_mpi",
    "compute_velocity_sf_reduction_3d_periodic_z_slab_mpi",
    "exchange_periodic_halo_z",
    "extract_local_z_slab",
)
