"""X-slab decomposition helpers for 2D MPI structure-function reductions."""

from __future__ import annotations

import numpy as np

from .slab_decomp_3d import compute_slab_bounds_1d


def _sum_and_count(values: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(values)
    return float(np.sum(values[valid], dtype=np.float64)), int(np.count_nonzero(valid))


def _exchange_neighbor_rows_axis0(local_arr: np.ndarray, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        return None, None

    left_halo = None
    right_halo = None
    if rank > 0:
        left_halo = np.empty((1, local_arr.shape[1]), dtype=local_arr.dtype)
        comm.Sendrecv(
            sendbuf=np.ascontiguousarray(local_arr[:1, :]),
            dest=rank - 1,
            sendtag=701,
            recvbuf=left_halo,
            source=rank - 1,
            recvtag=702,
        )
    if rank < size - 1:
        right_halo = np.empty((1, local_arr.shape[1]), dtype=local_arr.dtype)
        comm.Sendrecv(
            sendbuf=np.ascontiguousarray(local_arr[-1:, :]),
            dest=rank + 1,
            sendtag=702,
            recvbuf=right_halo,
            source=rank + 1,
            recvtag=701,
        )
    return left_halo, right_halo


def gradient_axis0_distributed_nonperiodic_2d(
    local_arr: np.ndarray, spacing: float, comm
) -> np.ndarray:
    """Match ``np.gradient(..., axis=0)`` on a globally decomposed first axis."""
    grad = np.empty_like(local_arr, dtype=np.float64)
    n_local = local_arr.shape[0]
    rank = comm.Get_rank()
    size = comm.Get_size()
    left_halo, right_halo = _exchange_neighbor_rows_axis0(local_arr, comm)

    if n_local > 2:
        grad[1:-1, :] = (local_arr[2:, :] - local_arr[:-2, :]) / (2.0 * spacing)

    if n_local == 1:
        if rank == 0 and size == 1:
            grad[0, :] = 0.0
        elif rank == 0:
            grad[0, :] = (right_halo[0, :] - local_arr[0, :]) / spacing
        elif rank == size - 1:
            grad[0, :] = (local_arr[0, :] - left_halo[0, :]) / spacing
        else:
            grad[0, :] = (right_halo[0, :] - left_halo[0, :]) / (2.0 * spacing)
        return grad

    if rank == 0:
        grad[0, :] = (local_arr[1, :] - local_arr[0, :]) / spacing
    else:
        grad[0, :] = (local_arr[1, :] - left_halo[0, :]) / (2.0 * spacing)

    if rank == size - 1:
        grad[-1, :] = (local_arr[-1, :] - local_arr[-2, :]) / spacing
    else:
        grad[-1, :] = (right_halo[0, :] - local_arr[-2, :]) / (2.0 * spacing)

    if n_local == 2 and rank > 0 and rank < size - 1:
        grad[0, :] = (local_arr[1, :] - left_halo[0, :]) / (2.0 * spacing)
        grad[1, :] = (right_halo[0, :] - local_arr[0, :]) / (2.0 * spacing)

    return grad


def _propagate_periodic_halo_axis0_2d(local_arr: np.ndarray, halo_depth: int, comm, direction: str):
    if halo_depth < 0:
        raise ValueError("halo_depth must be non-negative.")
    if halo_depth == 0:
        return np.empty((0, local_arr.shape[1]), dtype=local_arr.dtype)

    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        if direction == "forward":
            return np.ascontiguousarray(local_arr[:halo_depth, :])
        return np.ascontiguousarray(local_arr[-halo_depth:, :])

    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'.")

    local_take = min(halo_depth, local_arr.shape[0])
    front = np.ascontiguousarray(local_arr[:local_take, :])
    back = np.ascontiguousarray(local_arr[-local_take:, :])
    gathered = comm.allgather((front, back))

    pieces = []
    remaining = halo_depth
    if direction == "forward":
        rank_order = [((rank + step) % size) for step in range(1, size)]
        for other_rank in rank_order:
            if remaining <= 0:
                break
            other_front = gathered[other_rank][0]
            take = min(remaining, other_front.shape[0])
            if take > 0:
                pieces.append(np.ascontiguousarray(other_front[:take, :]))
                remaining -= take
    else:
        near_to_far = []
        rank_order = [((rank - step) % size) for step in range(1, size)]
        for other_rank in rank_order:
            if remaining <= 0:
                break
            other_back = gathered[other_rank][1]
            take = min(remaining, other_back.shape[0])
            if take > 0:
                near_to_far.append(np.ascontiguousarray(other_back[-take:, :]))
                remaining -= take
        pieces = list(reversed(near_to_far))

    if not pieces:
        return np.empty((0, local_arr.shape[1]), dtype=local_arr.dtype)
    return np.concatenate(pieces, axis=0)


def exchange_periodic_halo_axis0_2d(local_arr: np.ndarray, halo_depth: int, comm) -> np.ndarray:
    left_halo = _propagate_periodic_halo_axis0_2d(local_arr, halo_depth, comm, "backward")
    right_halo = _propagate_periodic_halo_axis0_2d(local_arr, halo_depth, comm, "forward")
    return np.concatenate((left_halo, local_arr, right_halo), axis=0)


def periodic_shift_axis0_2d(local_arr: np.ndarray, shift: int, comm) -> np.ndarray:
    if shift == 0:
        return np.ascontiguousarray(local_arr)
    extended = exchange_periodic_halo_axis0_2d(local_arr, shift, comm)
    core_start = shift
    core_stop = core_start + local_arr.shape[0]
    return extended[core_start + shift : core_stop + shift, :]


def _shift_array_nonperiodic_local_2d(local_arr: np.ndarray, shift: int, axis: int) -> np.ndarray:
    if shift == 0:
        return np.ascontiguousarray(local_arr)

    shifted = np.full(local_arr.shape, np.nan, dtype=np.result_type(local_arr, np.float64))
    dst = [slice(None), slice(None)]
    src = [slice(None), slice(None)]
    dst[axis] = slice(None, -shift)
    src[axis] = slice(shift, None)
    shifted[tuple(dst)] = local_arr[tuple(src)]
    return shifted


def _propagate_forward_halo_axis0_nonperiodic_2d(
    local_arr: np.ndarray, halo_depth: int, comm
) -> np.ndarray:
    if halo_depth < 0:
        raise ValueError("halo_depth must be non-negative.")
    if halo_depth == 0:
        return np.empty((0, local_arr.shape[1]), dtype=local_arr.dtype)

    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        return np.empty((0, local_arr.shape[1]), dtype=local_arr.dtype)

    prefix_count = min(halo_depth, local_arr.shape[0])
    prefix = np.ascontiguousarray(local_arr[:prefix_count, :])
    aggregate = np.empty((halo_depth, local_arr.shape[1]), dtype=local_arr.dtype)
    if prefix_count > 0:
        aggregate[:prefix_count, :] = prefix
    aggregate_count = prefix_count

    latest_count = 0
    latest = np.empty((halo_depth, local_arr.shape[1]), dtype=local_arr.dtype)

    for step in range(size - 1):
        count_tag = 900 + 2 * step
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
            latest[:incoming_count, :] = recvbuf[:incoming_count, :]

            combined_count = min(halo_depth, prefix_count + incoming_count)
            take_from_prefix = min(prefix_count, combined_count)
            if take_from_prefix > 0:
                aggregate[:take_from_prefix, :] = prefix[:take_from_prefix, :]
            take_from_incoming = combined_count - take_from_prefix
            if take_from_incoming > 0:
                aggregate[take_from_prefix:combined_count, :] = recvbuf[:take_from_incoming, :]
            aggregate_count = combined_count

    if latest_count == 0:
        return np.empty((0, local_arr.shape[1]), dtype=local_arr.dtype)
    return np.ascontiguousarray(latest[:latest_count, :])


def shift_axis0_nonperiodic_2d(local_arr: np.ndarray, shift: int, comm) -> np.ndarray:
    if shift == 0:
        return np.ascontiguousarray(local_arr)
    if comm.Get_size() == 1:
        return _shift_array_nonperiodic_local_2d(local_arr, shift, axis=0)

    right_halo = _propagate_forward_halo_axis0_nonperiodic_2d(local_arr, shift, comm)
    extended = np.concatenate((local_arr, right_halo), axis=0)
    shifted = np.full(local_arr.shape, np.nan, dtype=np.result_type(local_arr, np.float64))
    available = min(local_arr.shape[0], max(extended.shape[0] - shift, 0))
    if available > 0:
        shifted[:available, :] = extended[shift : shift + available, :]
    return shifted


def _shift_axis1(local_arr: np.ndarray, shift: int, periodic: bool) -> np.ndarray:
    if shift == 0:
        return np.ascontiguousarray(local_arr)
    if periodic:
        return np.roll(local_arr, shift=-shift, axis=1)
    return _shift_array_nonperiodic_local_2d(local_arr, shift, axis=1)


def _to_internal_x_slab_2d(local_arr: np.ndarray, layout: str) -> np.ndarray:
    if layout == "internal":
        return np.ascontiguousarray(local_arr)
    if layout == "public":
        return np.ascontiguousarray(local_arr.T)
    raise ValueError("layout must be 'public' or 'internal'.")


def calculate_advection_2d_public_x_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    scalar_local: np.ndarray | None = None,
    layout: str = "public",
    comm=None,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Distributed version of ``calculate_advection_2d`` on public x-slabs."""
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    dx = np.abs(x[0] - x[1])
    dy = np.abs(y[0] - y[1])
    u_int = _to_internal_x_slab_2d(u_local, layout)
    v_int = _to_internal_x_slab_2d(v_local, layout)

    if scalar_local is not None:
        s_int = _to_internal_x_slab_2d(scalar_local, layout)
        dsdx = gradient_axis0_distributed_nonperiodic_2d(s_int, dx, comm)
        dsdy = np.gradient(s_int, dy, axis=1)
        output = u_int * dsdx + v_int * dsdy
        return output if layout == "internal" else output.T

    dudx = gradient_axis0_distributed_nonperiodic_2d(u_int, dx, comm)
    dudy = np.gradient(u_int, dy, axis=1)
    dvdx = gradient_axis0_distributed_nonperiodic_2d(v_int, dx, comm)
    dvdy = np.gradient(v_int, dy, axis=1)
    adv_u = u_int * dudx + v_int * dudy
    adv_v = u_int * dvdx + v_int * dvdy
    if layout == "internal":
        return adv_u, adv_v
    return adv_u.T, adv_v.T


def compute_directional_sf_2d_public_x_slab_mpi(
    u_local: np.ndarray,
    v_local: np.ndarray,
    *,
    direction: str,
    shift: int,
    sf_type: tuple[str, ...],
    boundary: str | None = None,
    scalar_local: np.ndarray | None = None,
    adv_x_local: np.ndarray | None = None,
    adv_y_local: np.ndarray | None = None,
    adv_scalar_local: np.ndarray | None = None,
    layout: str = "public",
    comm=None,
) -> dict[str, float]:
    """Compute 2D structure functions on distributed public x-slabs."""
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    u_int = _to_internal_x_slab_2d(u_local, layout)
    v_int = _to_internal_x_slab_2d(v_local, layout)
    s_int = None if scalar_local is None else _to_internal_x_slab_2d(scalar_local, layout)
    adv_x_int = None if adv_x_local is None else _to_internal_x_slab_2d(adv_x_local, layout)
    adv_y_int = None if adv_y_local is None else _to_internal_x_slab_2d(adv_y_local, layout)
    adv_s_int = None if adv_scalar_local is None else _to_internal_x_slab_2d(adv_scalar_local, layout)

    if direction == "x":
        periodic = boundary in {"periodic-x", "periodic-all"}
        shift_fn = (
            (lambda arr: periodic_shift_axis0_2d(arr, shift, comm))
            if periodic
            else (lambda arr: shift_axis0_nonperiodic_2d(arr, shift, comm))
        )
        longitudinal = "u"
    elif direction == "y":
        periodic = boundary in {"periodic-y", "periodic-all"}
        shift_fn = lambda arr: _shift_axis1(arr, shift, periodic)
        longitudinal = "v"
    else:
        raise ValueError("direction must be 'x' or 'y'.")

    u_shift = shift_fn(u_int)
    v_shift = shift_fn(v_int)
    du = u_shift - u_int
    dv = v_shift - v_int

    if longitudinal == "u":
        d_long = du
        d_trans_sq = dv * dv
    else:
        d_long = dv
        d_trans_sq = du * du

    ds = None
    if s_int is not None:
        ds = shift_fn(s_int) - s_int

    local_reductions: dict[str, float | int] = {}
    if "ASF_V" in sf_type:
        adv_x_shift = shift_fn(adv_x_int)
        adv_y_shift = shift_fn(adv_y_int)
        values = (adv_x_shift - adv_x_int) * du + (adv_y_shift - adv_y_int) * dv
        local_reductions["SF_advection_velocity_sum"], local_reductions[
            "SF_advection_velocity_count"
        ] = _sum_and_count(values)
    if "ASF_S" in sf_type:
        adv_s_shift = shift_fn(adv_s_int)
        values = (adv_s_shift - adv_s_int) * ds
        local_reductions["SF_advection_scalar_sum"], local_reductions[
            "SF_advection_scalar_count"
        ] = _sum_and_count(values)
    if "LL" in sf_type:
        local_reductions["SF_LL_sum"], local_reductions["SF_LL_count"] = _sum_and_count(
            d_long**2
        )
    if "TT" in sf_type:
        local_reductions["SF_TT_sum"], local_reductions["SF_TT_count"] = _sum_and_count(
            d_trans_sq
        )
    if "SS" in sf_type:
        local_reductions["SF_SS_sum"], local_reductions["SF_SS_count"] = _sum_and_count(
            ds**2
        )
    if "LLL" in sf_type:
        local_reductions["SF_LLL_sum"], local_reductions["SF_LLL_count"] = _sum_and_count(
            d_long**3
        )
    if "LTT" in sf_type:
        local_reductions["SF_LTT_sum"], local_reductions["SF_LTT_count"] = _sum_and_count(
            d_long * d_trans_sq
        )
    if "LSS" in sf_type:
        local_reductions["SF_LSS_sum"], local_reductions["SF_LSS_count"] = _sum_and_count(
            d_long * ds**2
        )

    global_reductions = {
        key: comm.allreduce(value) for key, value in local_reductions.items()
    }
    output = {}
    for name in (
        "SF_advection_velocity",
        "SF_advection_scalar",
        "SF_LL",
        "SF_TT",
        "SF_SS",
        "SF_LLL",
        "SF_LTT",
        "SF_LSS",
    ):
        sum_key = f"{name}_sum"
        count_key = f"{name}_count"
        if sum_key not in global_reductions:
            continue
        count = int(global_reductions[count_key])
        output[name] = np.nan if count == 0 else float(global_reductions[sum_key]) / count
    return output


__all__ = (
    "calculate_advection_2d_public_x_slab_mpi",
    "compute_directional_sf_2d_public_x_slab_mpi",
    "compute_slab_bounds_1d",
)
