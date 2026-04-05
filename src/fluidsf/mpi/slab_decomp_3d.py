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
    """Collect periodic halo cells from successive ranks in one z direction.

    This gathers only the boundary slices each rank can contribute up to
    ``halo_depth``. It avoids the older hop-by-hop full-slab propagation,
    which paid one ``Sendrecv`` per slab hop for each requested z shift.
    """
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

    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'.")

    local_take = min(halo_depth, local_arr.shape[2])
    front = np.ascontiguousarray(local_arr[:, :, :local_take])
    back = np.ascontiguousarray(local_arr[:, :, -local_take:])
    gathered = comm.allgather((front, back))

    pieces = []
    remaining = halo_depth
    if direction == "forward":
        rank_order = [((rank + step) % size) for step in range(1, size)]
        for other_rank in rank_order:
            if remaining <= 0:
                break
            other_front = gathered[other_rank][0]
            take = min(remaining, other_front.shape[2])
            if take > 0:
                pieces.append(np.ascontiguousarray(other_front[:, :, :take]))
                remaining -= take
    else:
        near_to_far = []
        rank_order = [((rank - step) % size) for step in range(1, size)]
        for other_rank in rank_order:
            if remaining <= 0:
                break
            other_back = gathered[other_rank][1]
            take = min(remaining, other_back.shape[2])
            if take > 0:
                near_to_far.append(np.ascontiguousarray(other_back[:, :, -take:]))
                remaining -= take
        pieces = list(reversed(near_to_far))

    if not pieces:
        return np.empty(local_arr.shape[:2] + (0,), dtype=local_arr.dtype)
    return np.concatenate(pieces, axis=2)


def exchange_periodic_halo_z(local_arr: np.ndarray, halo_depth: int, comm) -> np.ndarray:
    """Extend a local z-slab with periodic halo cells on both sides."""
    left_halo = _propagate_periodic_halo_z(local_arr, halo_depth, comm, "backward")
    right_halo = _propagate_periodic_halo_z(local_arr, halo_depth, comm, "forward")
    return np.concatenate((left_halo, local_arr, right_halo), axis=2)


def _exchange_neighbor_planes_axis2(local_arr: np.ndarray, comm):
    """Exchange one non-periodic neighbor plane on the decomposed axis."""
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        return None, None

    left_halo = None
    right_halo = None
    if rank > 0:
        left_halo = np.empty(local_arr.shape[:2] + (1,), dtype=local_arr.dtype)
        comm.Sendrecv(
            sendbuf=np.ascontiguousarray(local_arr[:, :, :1]),
            dest=rank - 1,
            sendtag=11,
            recvbuf=left_halo,
            source=rank - 1,
            recvtag=22,
        )
    if rank < size - 1:
        right_halo = np.empty(local_arr.shape[:2] + (1,), dtype=local_arr.dtype)
        comm.Sendrecv(
            sendbuf=np.ascontiguousarray(local_arr[:, :, -1:]),
            dest=rank + 1,
            sendtag=22,
            recvbuf=right_halo,
            source=rank + 1,
            recvtag=11,
        )
    return left_halo, right_halo


def _exchange_neighbor_planes_axis0(local_arr: np.ndarray, comm):
    """Exchange one non-periodic neighbor plane on the first axis."""
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        return None, None

    left_halo = None
    right_halo = None
    if rank > 0:
        left_halo = np.empty((1,) + local_arr.shape[1:], dtype=local_arr.dtype)
        comm.Sendrecv(
            sendbuf=np.ascontiguousarray(local_arr[:1, :, :]),
            dest=rank - 1,
            sendtag=31,
            recvbuf=left_halo,
            source=rank - 1,
            recvtag=42,
        )
    if rank < size - 1:
        right_halo = np.empty((1,) + local_arr.shape[1:], dtype=local_arr.dtype)
        comm.Sendrecv(
            sendbuf=np.ascontiguousarray(local_arr[-1:, :, :]),
            dest=rank + 1,
            sendtag=42,
            recvbuf=right_halo,
            source=rank + 1,
            recvtag=31,
        )
    return left_halo, right_halo


def _sum_and_count(values: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(values)
    return float(np.sum(values[valid], dtype=np.float64)), int(np.count_nonzero(valid))


def _valid_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.isfinite(arrays[0])
    for arr in arrays[1:]:
        mask &= np.isfinite(arr)
    return mask


def _sum_and_count_square(values: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(values)
    selected = values[valid]
    return float(np.sum(selected * selected, dtype=np.float64)), int(np.count_nonzero(valid))


def _sum_and_count_cube(values: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(values)
    selected = values[valid]
    return float(np.sum(selected * selected * selected, dtype=np.float64)), int(
        np.count_nonzero(valid)
    )


def _sum_and_count_product(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    valid = _valid_mask(a, b)
    a_sel = a[valid]
    b_sel = b[valid]
    return float(np.sum(a_sel * b_sel, dtype=np.float64)), int(np.count_nonzero(valid))


def _sum_and_count_sum_squares(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    valid = _valid_mask(a, b)
    a_sel = a[valid]
    b_sel = b[valid]
    return float(np.sum(a_sel * a_sel + b_sel * b_sel, dtype=np.float64)), int(
        np.count_nonzero(valid)
    )


def _sum_and_count_longitudinal_transverse(
    d_long: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, int]:
    valid = _valid_mask(d_long, a, b)
    long_sel = d_long[valid]
    a_sel = a[valid]
    b_sel = b[valid]
    return float(
        np.sum(long_sel * (a_sel * a_sel + b_sel * b_sel), dtype=np.float64)
    ), int(np.count_nonzero(valid))


def _sum_and_count_longitudinal_square(
    d_long: np.ndarray, values: np.ndarray
) -> tuple[float, int]:
    valid = _valid_mask(d_long, values)
    long_sel = d_long[valid]
    value_sel = values[valid]
    return float(np.sum(long_sel * value_sel * value_sel, dtype=np.float64)), int(
        np.count_nonzero(valid)
    )


def _sum_and_count_advective_velocity(
    d_adv_x: np.ndarray,
    du: np.ndarray,
    d_adv_y: np.ndarray,
    dv: np.ndarray,
    d_adv_z: np.ndarray,
    dw: np.ndarray,
) -> tuple[float, int]:
    valid = _valid_mask(d_adv_x, du, d_adv_y, dv, d_adv_z, dw)
    adv_x_sel = d_adv_x[valid]
    du_sel = du[valid]
    adv_y_sel = d_adv_y[valid]
    dv_sel = dv[valid]
    adv_z_sel = d_adv_z[valid]
    dw_sel = dw[valid]
    return float(
        np.sum(
            adv_x_sel * du_sel + adv_y_sel * dv_sel + adv_z_sel * dw_sel,
            dtype=np.float64,
        )
    ), int(np.count_nonzero(valid))


def _finalize_global_reductions_mpi(local_reductions, sf_type, comm) -> dict[str, float]:
    global_reductions = {
        key: comm.allreduce(value) for key, value in local_reductions.items()
    }
    return finalize_structure_function_reduction(global_reductions, sf_type)


def _require_mpi():
    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - exercised where mpi4py is missing
        raise ImportError(
            "mpi4py is required for the slab-decomposition MPI backend."
        ) from exc
    return MPI


def gradient_axis2_distributed_nonperiodic(local_arr: np.ndarray, spacing: float, comm) -> np.ndarray:
    """Match ``np.gradient(..., axis=2)`` on a globally decomposed axis."""
    grad = np.empty_like(local_arr, dtype=np.float64)
    n_local = local_arr.shape[2]
    rank = comm.Get_rank()
    size = comm.Get_size()
    left_halo, right_halo = _exchange_neighbor_planes_axis2(local_arr, comm)

    if n_local > 2:
        grad[:, :, 1:-1] = (local_arr[:, :, 2:] - local_arr[:, :, :-2]) / (2.0 * spacing)

    if n_local == 1:
        if rank == 0 and size == 1:
            grad[:, :, 0] = 0.0
        elif rank == 0:
            grad[:, :, 0] = (right_halo[:, :, 0] - local_arr[:, :, 0]) / spacing
        elif rank == size - 1:
            grad[:, :, 0] = (local_arr[:, :, 0] - left_halo[:, :, 0]) / spacing
        else:
            grad[:, :, 0] = (right_halo[:, :, 0] - left_halo[:, :, 0]) / (2.0 * spacing)
        return grad

    if rank == 0:
        grad[:, :, 0] = (local_arr[:, :, 1] - local_arr[:, :, 0]) / spacing
    else:
        grad[:, :, 0] = (local_arr[:, :, 1] - left_halo[:, :, 0]) / (2.0 * spacing)

    if rank == size - 1:
        grad[:, :, -1] = (local_arr[:, :, -1] - local_arr[:, :, -2]) / spacing
    else:
        grad[:, :, -1] = (right_halo[:, :, 0] - local_arr[:, :, -2]) / (2.0 * spacing)

    if n_local == 2 and rank > 0 and rank < size - 1:
        grad[:, :, 0] = (local_arr[:, :, 1] - left_halo[:, :, 0]) / (2.0 * spacing)
        grad[:, :, 1] = (right_halo[:, :, 0] - local_arr[:, :, 0]) / (2.0 * spacing)

    return grad


def gradient_axis0_distributed_nonperiodic(local_arr: np.ndarray, spacing: float, comm) -> np.ndarray:
    """Match ``np.gradient(..., axis=0)`` on a globally decomposed first axis."""
    grad = np.empty_like(local_arr, dtype=np.float64)
    n_local = local_arr.shape[0]
    rank = comm.Get_rank()
    size = comm.Get_size()
    left_halo, right_halo = _exchange_neighbor_planes_axis0(local_arr, comm)

    if n_local > 2:
        grad[1:-1, :, :] = (local_arr[2:, :, :] - local_arr[:-2, :, :]) / (2.0 * spacing)

    if n_local == 1:
        if rank == 0 and size == 1:
            grad[0, :, :] = 0.0
        elif rank == 0:
            grad[0, :, :] = (right_halo[0, :, :] - local_arr[0, :, :]) / spacing
        elif rank == size - 1:
            grad[0, :, :] = (local_arr[0, :, :] - left_halo[0, :, :]) / spacing
        else:
            grad[0, :, :] = (right_halo[0, :, :] - left_halo[0, :, :]) / (2.0 * spacing)
        return grad

    if rank == 0:
        grad[0, :, :] = (local_arr[1, :, :] - local_arr[0, :, :]) / spacing
    else:
        grad[0, :, :] = (local_arr[1, :, :] - left_halo[0, :, :]) / (2.0 * spacing)

    if rank == size - 1:
        grad[-1, :, :] = (local_arr[-1, :, :] - local_arr[-2, :, :]) / spacing
    else:
        grad[-1, :, :] = (right_halo[0, :, :] - local_arr[-2, :, :]) / (2.0 * spacing)

    if n_local == 2 and rank > 0 and rank < size - 1:
        grad[0, :, :] = (local_arr[1, :, :] - left_halo[0, :, :]) / (2.0 * spacing)
        grad[1, :, :] = (right_halo[0, :, :] - local_arr[0, :, :]) / (2.0 * spacing)

    return grad


def calculate_advection_3d_z_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    w_local: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    scalar_local: np.ndarray | None = None,
    comm=None,
):
    """Distributed version of ``calculate_advection_3d`` on internal ``(z, y, x)`` slabs."""
    if comm is None:
        _require_mpi()
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    dx = np.abs(x[0] - x[1])
    dy = np.abs(y[0] - y[1])
    dz = np.abs(z[0] - z[1])

    if scalar_local is not None:
        dsdx = gradient_axis2_distributed_nonperiodic(scalar_local, dx, comm)
        dsdy = np.gradient(scalar_local, dy, axis=1)
        dsdz = np.gradient(scalar_local, dz, axis=0)
        return u_local * dsdx + v_local * dsdy + w_local * dsdz

    dudx = gradient_axis2_distributed_nonperiodic(u_local, dx, comm)
    dudy = np.gradient(u_local, dy, axis=1)
    dudz = np.gradient(u_local, dz, axis=0)

    dvdx = gradient_axis2_distributed_nonperiodic(v_local, dx, comm)
    dvdy = np.gradient(v_local, dy, axis=1)
    dvdz = np.gradient(v_local, dz, axis=0)

    dwdx = gradient_axis2_distributed_nonperiodic(w_local, dx, comm)
    dwdy = np.gradient(w_local, dy, axis=1)
    dwdz = np.gradient(w_local, dz, axis=0)

    u_advection = u_local * dudx + v_local * dudy + w_local * dudz
    v_advection = u_local * dvdx + v_local * dvdy + w_local * dvdz
    w_advection = u_local * dwdx + v_local * dwdy + w_local * dwdz
    return u_advection, v_advection, w_advection


def _propagate_periodic_halo_axis0(local_arr: np.ndarray, halo_depth: int, comm, direction: str):
    """Collect periodic halo cells from successive ranks on the first axis."""
    if halo_depth < 0:
        raise ValueError("halo_depth must be non-negative.")
    if halo_depth == 0:
        return np.empty((0,) + local_arr.shape[1:], dtype=local_arr.dtype)

    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        if direction == "forward":
            return np.ascontiguousarray(local_arr[:halo_depth, :, :])
        return np.ascontiguousarray(local_arr[-halo_depth:, :, :])

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
            sendtag=hop + 100,
            recvbuf=recvbuf,
            source=source,
            recvtag=hop + 100,
        )
        take = min(remaining, recvbuf.shape[0])
        if direction == "forward":
            pieces.append(np.ascontiguousarray(recvbuf[:take, :, :]))
        else:
            pieces.append(np.ascontiguousarray(recvbuf[-take:, :, :]))
        remaining -= take
        propagated = recvbuf
        hop += 1

    if direction == "forward":
        return np.concatenate(pieces, axis=0)
    pieces.reverse()
    return np.concatenate(pieces, axis=0)


def exchange_periodic_halo_axis0(local_arr: np.ndarray, halo_depth: int, comm) -> np.ndarray:
    """Extend a local x-slab with periodic halo cells on both sides."""
    left_halo = _propagate_periodic_halo_axis0(local_arr, halo_depth, comm, "backward")
    right_halo = _propagate_periodic_halo_axis0(local_arr, halo_depth, comm, "forward")
    return np.concatenate((left_halo, local_arr, right_halo), axis=0)


def periodic_shift_axis0(local_arr: np.ndarray, shift: int, comm) -> np.ndarray:
    """Shift the decomposed first axis periodically by ``shift`` cells."""
    if shift == 0:
        return np.ascontiguousarray(local_arr)
    extended = exchange_periodic_halo_axis0(local_arr, shift, comm)
    core_start = shift
    core_stop = core_start + local_arr.shape[0]
    return extended[core_start + shift : core_stop + shift, :, :]


def _shift_array_nonperiodic_local(
    local_arr: np.ndarray, shift: int, axis: int
) -> np.ndarray:
    """Shift a non-decomposed axis forward with NaN padding."""
    if shift == 0:
        return np.ascontiguousarray(local_arr)

    shifted = np.full(local_arr.shape, np.nan, dtype=np.result_type(local_arr, np.float64))
    dst = [slice(None), slice(None), slice(None)]
    src = [slice(None), slice(None), slice(None)]
    dst[axis] = slice(None, -shift)
    src[axis] = slice(shift, None)
    shifted[tuple(dst)] = local_arr[tuple(src)]
    return shifted


def _propagate_forward_halo_axis0_nonperiodic(
    local_arr: np.ndarray, halo_depth: int, comm
) -> np.ndarray:
    """Collect forward halo cells from higher ranks without wraparound.

    The halo is assembled by forwarding only the capped prefix slices lower
    ranks still need. This avoids the previous ``allgather`` that broadcast
    every rank's prefix to every other rank.
    """
    if halo_depth < 0:
        raise ValueError("halo_depth must be non-negative.")
    if halo_depth == 0:
        return np.empty((0,) + local_arr.shape[1:], dtype=local_arr.dtype)

    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        return np.empty((0,) + local_arr.shape[1:], dtype=local_arr.dtype)

    prefix_count = min(halo_depth, local_arr.shape[0])
    prefix = np.ascontiguousarray(local_arr[:prefix_count, :, :])
    aggregate = np.empty((halo_depth,) + local_arr.shape[1:], dtype=local_arr.dtype)
    if prefix_count > 0:
        aggregate[:prefix_count, :, :] = prefix
    aggregate_count = prefix_count

    latest_count = 0
    latest = np.empty((halo_depth,) + local_arr.shape[1:], dtype=local_arr.dtype)

    for step in range(size - 1):
        count_tag = 500 + 2 * step
        data_tag = count_tag + 1

        incoming_count = 0
        if rank > 0 and rank < size - 1:
            send_count = np.array([aggregate_count], dtype=np.int32)
            recv_count = np.zeros(1, dtype=np.int32)
            comm.Sendrecv(
                sendbuf=send_count,
                dest=rank - 1,
                sendtag=count_tag,
                recvbuf=recv_count,
                source=rank + 1,
                recvtag=count_tag,
            )
            incoming_count = int(recv_count[0])
            recvbuf = np.empty_like(aggregate)
            comm.Sendrecv(
                sendbuf=aggregate,
                dest=rank - 1,
                sendtag=data_tag,
                recvbuf=recvbuf,
                source=rank + 1,
                recvtag=data_tag,
            )
        elif rank > 0:
            send_count = np.array([aggregate_count], dtype=np.int32)
            comm.Send(send_count, dest=rank - 1, tag=count_tag)
            comm.Send(aggregate, dest=rank - 1, tag=data_tag)
            recvbuf = None
        elif rank < size - 1:
            recv_count = np.zeros(1, dtype=np.int32)
            comm.Recv(recv_count, source=rank + 1, tag=count_tag)
            incoming_count = int(recv_count[0])
            recvbuf = np.empty_like(aggregate)
            comm.Recv(recvbuf, source=rank + 1, tag=data_tag)
        else:
            recvbuf = None

        if rank < size - 1 and incoming_count > 0:
            latest_count = incoming_count
            latest[:incoming_count, :, :] = recvbuf[:incoming_count, :, :]

            combined_count = min(halo_depth, prefix_count + incoming_count)
            take_from_prefix = min(prefix_count, combined_count)
            if take_from_prefix > 0:
                aggregate[:take_from_prefix, :, :] = prefix[:take_from_prefix, :, :]
            take_from_incoming = combined_count - take_from_prefix
            if take_from_incoming > 0:
                aggregate[take_from_prefix:combined_count, :, :] = recvbuf[
                    :take_from_incoming, :, :
                ]
            aggregate_count = combined_count

    if latest_count == 0:
        return np.empty((0,) + local_arr.shape[1:], dtype=local_arr.dtype)
    return np.ascontiguousarray(latest[:latest_count, :, :])


def shift_axis0_nonperiodic(local_arr: np.ndarray, shift: int, comm) -> np.ndarray:
    """Shift the decomposed first axis forward with NaN padding."""
    if shift == 0:
        return np.ascontiguousarray(local_arr)
    if comm.Get_size() == 1:
        return _shift_array_nonperiodic_local(local_arr, shift, axis=0)

    right_halo = _propagate_forward_halo_axis0_nonperiodic(local_arr, shift, comm)
    extended = np.concatenate((local_arr, right_halo), axis=0)
    shifted = np.full(local_arr.shape, np.nan, dtype=np.result_type(local_arr, np.float64))
    available = min(local_arr.shape[0], max(extended.shape[0] - shift, 0))
    if available > 0:
        shifted[:available, :, :] = extended[shift : shift + available, :, :]
    return shifted


def compute_directional_sf_3d_public_x_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    w_local: np.ndarray,
    *,
    direction: str,
    shift: int,
    sf_type: tuple[str, ...],
    boundary: str | None = None,
    scalar_local: np.ndarray | None = None,
    adv_x_local: np.ndarray | None = None,
    adv_y_local: np.ndarray | None = None,
    adv_z_local: np.ndarray | None = None,
    adv_scalar_local: np.ndarray | None = None,
    comm=None,
) -> dict[str, float]:
    """Compute axis-aligned 3D structure functions on public-layout x-owned slabs.

    The public MPI backend accepts arrays shaped ``(local_x, y, z)`` on each rank.
    To remain bit-for-bit compatible with the legacy serial 3D implementation, the
    directional shifts in this helper follow that legacy axis convention:
    ``x -> axis 2``, ``y -> axis 1``, ``z -> axis 0``.
    """
    if comm is None:
        _require_mpi()
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    is_periodic = boundary == "periodic-all"
    if boundary is None:
        if direction == "x":
            shift_fn = lambda arr: _shift_array_nonperiodic_local(arr, shift, axis=2)
            longitudinal = "u"
        elif direction == "y":
            shift_fn = lambda arr: _shift_array_nonperiodic_local(arr, shift, axis=1)
            longitudinal = "v"
        elif direction == "z":
            shift_fn = lambda arr: shift_axis0_nonperiodic(arr, shift, comm)
            longitudinal = "w"
        else:
            raise ValueError("direction must be one of 'x', 'y', or 'z'.")
    elif is_periodic:
        if direction == "x":
            shift_fn = lambda arr: np.roll(arr, shift=-shift, axis=2)
            longitudinal = "u"
        elif direction == "y":
            shift_fn = lambda arr: np.roll(arr, shift=-shift, axis=1)
            longitudinal = "v"
        elif direction == "z":
            shift_fn = lambda arr: periodic_shift_axis0(arr, shift, comm)
            longitudinal = "w"
        else:
            raise ValueError("direction must be one of 'x', 'y', or 'z'.")
    else:
        raise ValueError("boundary must be None or 'periodic-all'.")

    u_shift = shift_fn(u_local)
    v_shift = shift_fn(v_local)
    w_shift = shift_fn(w_local)
    du = u_shift - u_local
    dv = v_shift - v_local
    dw = w_shift - w_local

    if longitudinal == "u":
        d_long = du
        trans_a = dv
        trans_b = dw
    elif longitudinal == "v":
        d_long = dv
        trans_a = du
        trans_b = dw
    else:
        d_long = dw
        trans_a = du
        trans_b = dv

    ds = None
    if scalar_local is not None and any(name in sf_type for name in ("ASF_S", "SS", "LSS")):
        scalar_shift = shift_fn(scalar_local)
        ds = scalar_shift - scalar_local

    local_reductions: dict[str, float | int] = {}
    if "ASF_V" in sf_type:
        adv_x_shift = shift_fn(adv_x_local)
        adv_y_shift = shift_fn(adv_y_local)
        adv_z_shift = shift_fn(adv_z_local)
        local_reductions["SF_advection_velocity_sum"], local_reductions[
            "SF_advection_velocity_count"
        ] = _sum_and_count_advective_velocity(
            adv_x_shift - adv_x_local,
            du,
            adv_y_shift - adv_y_local,
            dv,
            adv_z_shift - adv_z_local,
            dw,
        )
    if "ASF_S" in sf_type:
        adv_scalar_shift = shift_fn(adv_scalar_local)
        local_reductions["SF_advection_scalar_sum"], local_reductions[
            "SF_advection_scalar_count"
        ] = _sum_and_count_product(adv_scalar_shift - adv_scalar_local, ds)
    if "LL" in sf_type:
        local_reductions["SF_LL_sum"], local_reductions["SF_LL_count"] = _sum_and_count_square(
            d_long
        )
    if "TT" in sf_type:
        local_reductions["SF_TT_sum"], local_reductions["SF_TT_count"] = _sum_and_count_sum_squares(
            trans_a, trans_b
        )
    if "SS" in sf_type:
        local_reductions["SF_SS_sum"], local_reductions["SF_SS_count"] = _sum_and_count_square(
            ds
        )
    if "LLL" in sf_type:
        local_reductions["SF_LLL_sum"], local_reductions["SF_LLL_count"] = _sum_and_count_cube(
            d_long
        )
    if "LTT" in sf_type:
        local_reductions["SF_LTT_sum"], local_reductions[
            "SF_LTT_count"
        ] = _sum_and_count_longitudinal_transverse(d_long, trans_a, trans_b)
    if "LSS" in sf_type:
        local_reductions["SF_LSS_sum"], local_reductions[
            "SF_LSS_count"
        ] = _sum_and_count_longitudinal_square(d_long, ds)

    global_reductions = {
        key: comm.allreduce(value) for key, value in local_reductions.items()
    }
    output = {}
    for name in ("SF_advection_velocity", "SF_advection_scalar", "SF_LL", "SF_TT", "SF_SS", "SF_LLL", "SF_LTT", "SF_LSS"):
        sum_key = f"{name}_sum"
        count_key = f"{name}_count"
        if sum_key not in global_reductions:
            continue
        count = int(global_reductions[count_key])
        output[name] = np.nan if count == 0 else float(global_reductions[sum_key]) / count
    return output


def calculate_advection_3d_public_x_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    w_local: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    scalar_local: np.ndarray | None = None,
    comm=None,
):
    """Distributed version of ``calculate_advection_3d`` on public slabs.

    The input slabs are shaped ``(local_x, y, z)``, but the gradient directions
    intentionally mirror the legacy serial 3D implementation:
    ``x -> axis 2``, ``y -> axis 1``, ``z -> axis 0``.
    """
    if comm is None:
        _require_mpi()
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    dx = np.abs(x[0] - x[1])
    dy = np.abs(y[0] - y[1])
    dz = np.abs(z[0] - z[1])

    if scalar_local is not None:
        dsdx = np.gradient(scalar_local, dx, axis=2)
        dsdy = np.gradient(scalar_local, dy, axis=1)
        dsdz = gradient_axis0_distributed_nonperiodic(scalar_local, dz, comm)
        return u_local * dsdx + v_local * dsdy + w_local * dsdz

    dudx = np.gradient(u_local, dx, axis=2)
    dudy = np.gradient(u_local, dy, axis=1)
    dudz = gradient_axis0_distributed_nonperiodic(u_local, dz, comm)

    dvdx = np.gradient(v_local, dx, axis=2)
    dvdy = np.gradient(v_local, dy, axis=1)
    dvdz = gradient_axis0_distributed_nonperiodic(v_local, dz, comm)

    dwdx = np.gradient(w_local, dx, axis=2)
    dwdy = np.gradient(w_local, dy, axis=1)
    dwdz = gradient_axis0_distributed_nonperiodic(w_local, dz, comm)

    u_advection = u_local * dudx + v_local * dudy + w_local * dudz
    v_advection = u_local * dvdx + v_local * dvdy + w_local * dvdz
    w_advection = u_local * dwdx + v_local * dwdy + w_local * dwdz
    return u_advection, v_advection, w_advection


def compute_advective_sf_direction_3d_public_x_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    w_local: np.ndarray,
    *,
    direction: str,
    shift: int,
    adv_x_local: np.ndarray | None = None,
    adv_y_local: np.ndarray | None = None,
    adv_z_local: np.ndarray | None = None,
    scalar_local: np.ndarray | None = None,
    adv_scalar_local: np.ndarray | None = None,
    sf_type: tuple[str, ...],
    comm=None,
) -> dict[str, float]:
    """Compute periodic advective SFs on public-layout x-owned slabs.

    This helper is currently internal-only. It assumes arrays shaped
    ``(local_x, y, z)``, but its shift directions intentionally mirror the
    legacy serial 3D implementation: ``x -> axis 2``, ``y -> axis 1``,
    ``z -> axis 0``.
    """
    if comm is None:
        _require_mpi()
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    if direction == "x":
        shift_fn = lambda arr: np.roll(arr, shift=-shift, axis=2)
    elif direction == "y":
        shift_fn = lambda arr: np.roll(arr, shift=-shift, axis=1)
    elif direction == "z":
        shift_fn = lambda arr: periodic_shift_axis0(arr, shift, comm)
    else:
        raise ValueError("direction must be one of 'x', 'y', or 'z'.")

    local_reductions = {}
    if "ASF_V" in sf_type:
        u_shift = shift_fn(u_local)
        v_shift = shift_fn(v_local)
        w_shift = shift_fn(w_local)
        adv_x_shift = shift_fn(adv_x_local)
        adv_y_shift = shift_fn(adv_y_local)
        adv_z_shift = shift_fn(adv_z_local)
        values = (
            (adv_x_shift - adv_x_local) * (u_shift - u_local)
            + (adv_y_shift - adv_y_local) * (v_shift - v_local)
            + (adv_z_shift - adv_z_local) * (w_shift - w_local)
        )
        local_reductions["SF_advection_velocity_sum"], local_reductions[
            "SF_advection_velocity_count"
        ] = _sum_and_count(values)
    if "ASF_S" in sf_type:
        scalar_shift = shift_fn(scalar_local)
        adv_scalar_shift = shift_fn(adv_scalar_local)
        values = (adv_scalar_shift - adv_scalar_local) * (scalar_shift - scalar_local)
        local_reductions["SF_advection_scalar_sum"], local_reductions[
            "SF_advection_scalar_count"
        ] = _sum_and_count(values)

    global_reductions = {
        key: comm.allreduce(value) for key, value in local_reductions.items()
    }
    output = {}
    if "ASF_V" in sf_type:
        count = int(global_reductions["SF_advection_velocity_count"])
        output["SF_advection_velocity"] = (
            np.nan
            if count == 0
            else float(global_reductions["SF_advection_velocity_sum"]) / count
        )
    if "ASF_S" in sf_type:
        count = int(global_reductions["SF_advection_scalar_count"])
        output["SF_advection_scalar"] = (
            np.nan if count == 0 else float(global_reductions["SF_advection_scalar_sum"]) / count
        )
    return output


def periodic_shift_axis2(local_arr: np.ndarray, shift: int, comm) -> np.ndarray:
    """Shift the decomposed axis periodically by ``shift`` cells."""
    if shift == 0:
        return np.ascontiguousarray(local_arr)
    extended = exchange_periodic_halo_z(local_arr, shift, comm)
    core_start = shift
    core_stop = core_start + local_arr.shape[2]
    return extended[:, :, core_start + shift : core_stop + shift]


def compute_advective_sf_direction_3d_periodic_z_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    w_local: np.ndarray,
    *,
    direction: str,
    shift: int,
    adv_x_local: np.ndarray | None = None,
    adv_y_local: np.ndarray | None = None,
    adv_z_local: np.ndarray | None = None,
    scalar_local: np.ndarray | None = None,
    adv_scalar_local: np.ndarray | None = None,
    sf_type: tuple[str, ...],
    comm=None,
) -> dict[str, float]:
    """Compute periodic directional advective structure functions on z-slabs."""
    if comm is None:
        _require_mpi()
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    if direction == "x":
        shift_fn = lambda arr: periodic_shift_axis2(arr, shift, comm)
    elif direction == "y":
        shift_fn = lambda arr: np.roll(arr, shift=-shift, axis=1)
    elif direction == "z":
        shift_fn = lambda arr: np.roll(arr, shift=-shift, axis=0)
    else:
        raise ValueError("direction must be one of 'x', 'y', or 'z'.")

    local_reductions = {}
    if "ASF_V" in sf_type:
        u_shift = shift_fn(u_local)
        v_shift = shift_fn(v_local)
        w_shift = shift_fn(w_local)
        adv_x_shift = shift_fn(adv_x_local)
        adv_y_shift = shift_fn(adv_y_local)
        adv_z_shift = shift_fn(adv_z_local)
        values = (
            (adv_x_shift - adv_x_local) * (u_shift - u_local)
            + (adv_y_shift - adv_y_local) * (v_shift - v_local)
            + (adv_z_shift - adv_z_local) * (w_shift - w_local)
        )
        local_reductions["SF_advection_velocity_sum"], local_reductions[
            "SF_advection_velocity_count"
        ] = _sum_and_count(values)
    if "ASF_S" in sf_type:
        scalar_shift = shift_fn(scalar_local)
        adv_scalar_shift = shift_fn(adv_scalar_local)
        values = (adv_scalar_shift - adv_scalar_local) * (scalar_shift - scalar_local)
        local_reductions["SF_advection_scalar_sum"], local_reductions[
            "SF_advection_scalar_count"
        ] = _sum_and_count(values)

    global_reductions = {
        key: comm.allreduce(value) for key, value in local_reductions.items()
    }
    output = {}
    if "ASF_V" in sf_type:
        count = int(global_reductions["SF_advection_velocity_count"])
        output["SF_advection_velocity"] = (
            np.nan
            if count == 0
            else float(global_reductions["SF_advection_velocity_sum"]) / count
        )
    if "ASF_S" in sf_type:
        count = int(global_reductions["SF_advection_scalar_count"])
        output["SF_advection_scalar"] = (
            np.nan if count == 0 else float(global_reductions["SF_advection_scalar_sum"]) / count
        )
    return output


def _local_velocity_sf_reduction_3d_periodic_from_extended(
    base_u: np.ndarray,
    base_v: np.ndarray,
    base_w: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    target_w: np.ndarray,
    shift_x: int,
    shift_y: int,
    shift_z: int,
    sf_type: tuple[str, ...],
) -> dict[str, float | int]:
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
    return local_reductions


def _local_scalar_sf_reduction_3d_periodic_from_extended(
    base_u: np.ndarray,
    base_v: np.ndarray,
    base_w: np.ndarray,
    base_s: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    target_w: np.ndarray,
    target_s: np.ndarray,
    shift_x: int,
    shift_y: int,
    shift_z: int,
    sf_type: tuple[str, ...],
) -> dict[str, float | int]:
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
    return local_reductions


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

    local_reductions = _local_velocity_sf_reduction_3d_periodic_from_extended(
        base_u,
        base_v,
        base_w,
        target_u,
        target_v,
        target_w,
        shift_x,
        shift_y,
        shift_z,
        sf_type,
    )
    return _finalize_global_reductions_mpi(local_reductions, sf_type, comm)


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

    local_reductions = _local_scalar_sf_reduction_3d_periodic_from_extended(
        base_u,
        base_v,
        base_w,
        base_s,
        target_u,
        target_v,
        target_w,
        target_s,
        shift_x,
        shift_y,
        shift_z,
        sf_type,
    )
    return _finalize_global_reductions_mpi(local_reductions, sf_type, comm)


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
    """Generate a periodic 3D SF grid from internal ``(x, y, local_z)`` slabs.

    This helper is intentionally a z-slab/internal-layout routine. Public MPI
    callers with distributed ``(local_x, y, z)`` inputs should use
    ``generate_structure_functions_3d(..., backend="mpi")`` instead.
    """
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

    global_nz = comm.allreduce(u_local.shape[2])
    if global_nz != len(z):
        raise ValueError(
            "The local z-slab sizes do not sum to len(z). "
            "Expected internal slabs shaped (len(x), len(y), local_z)."
        )
    if u_local.shape[0] != len(x) or u_local.shape[1] != len(y):
        raise ValueError(
            "Local slab x/y dimensions must match len(x) and len(y). "
            "Expected internal slabs shaped (len(x), len(y), local_z)."
        )
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

    velocity_sf_type = tuple(name for name in sf_type if name in ("LL", "TT", "LLL", "LTT"))
    scalar_sf_type = tuple(name for name in sf_type if name in ("SS", "LSS"))

    for shift_z in range(nz_half):
        halo_depth = int(shift_z)
        u_ext = exchange_periodic_halo_z(u_local, halo_depth, comm)
        v_ext = exchange_periodic_halo_z(v_local, halo_depth, comm)
        w_ext = exchange_periodic_halo_z(w_local, halo_depth, comm)
        s_ext = None
        if scalar_sf_type:
            s_ext = exchange_periodic_halo_z(scalar_local, halo_depth, comm)

        core_start = halo_depth
        core_stop = core_start + u_local.shape[2]
        base_u = u_ext[:, :, core_start:core_stop]
        base_v = v_ext[:, :, core_start:core_stop]
        base_w = w_ext[:, :, core_start:core_stop]
        target_u = u_ext[:, :, core_start + shift_z : core_stop + shift_z]
        target_v = v_ext[:, :, core_start + shift_z : core_stop + shift_z]
        target_w = w_ext[:, :, core_start + shift_z : core_stop + shift_z]
        base_s = None
        target_s = None
        if scalar_sf_type:
            base_s = s_ext[:, :, core_start:core_stop]
            target_s = s_ext[:, :, core_start + shift_z : core_stop + shift_z]

        for shift_x in range(nx_half):
            for shift_y in range(ny_half):
                if shift_x == 0 and shift_y == 0 and shift_z == 0:
                    continue
                reduced = {}
                if velocity_sf_type:
                    reduced.update(
                        _finalize_global_reductions_mpi(
                            _local_velocity_sf_reduction_3d_periodic_from_extended(
                                base_u,
                                base_v,
                                base_w,
                                target_u,
                                target_v,
                                target_w,
                                shift_x,
                                shift_y,
                                shift_z,
                                velocity_sf_type,
                            ),
                            velocity_sf_type,
                            comm,
                        )
                    )
                if scalar_sf_type:
                    reduced.update(
                        _finalize_global_reductions_mpi(
                            _local_scalar_sf_reduction_3d_periodic_from_extended(
                                base_u,
                                base_v,
                                base_w,
                                base_s,
                                target_u,
                                target_v,
                                target_w,
                                target_s,
                                shift_x,
                                shift_y,
                                shift_z,
                                scalar_sf_type,
                            ),
                            scalar_sf_type,
                            comm,
                        )
                    )
                for name in sf_type:
                    output[f"SF_{name}_grid"][shift_x, shift_y, shift_z] = reduced[
                        f"SF_{name}"
                    ]

    return output


__all__ = (
    "compute_slab_bounds_1d",
    "compute_scalar_sf_reduction_3d_periodic_z_slab_mpi",
    "generate_sf_grid_3d_periodic_z_slab_mpi",
    "compute_velocity_sf_reduction_3d_periodic_z_slab_mpi",
    "exchange_periodic_halo_z",
    "extract_local_z_slab",
)
