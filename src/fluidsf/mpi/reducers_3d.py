"""Local reduction kernels for 3D MPI structure-function calculations."""

from __future__ import annotations

import numpy as np


def shifted_difference_3d(
    arr: np.ndarray,
    shift_x: int,
    shift_y: int,
    shift_z: int,
    boundary=None,
) -> np.ndarray:
    """Return the increment field for a single displacement vector."""
    if boundary is not None:
        if "periodic-all" not in boundary:
            raise ValueError(
                "3D MPI reducers currently support only boundary=None or "
                "boundary='periodic-all'."
            )
        return np.roll(arr, shift=(-shift_x, -shift_y, -shift_z), axis=(0, 1, 2)) - arr

    src_shifted = (
        slice(shift_x, None),
        slice(shift_y, None),
        slice(shift_z, None),
    )
    src_base = (
        slice(None if shift_x == 0 else -shift_x),
        slice(None if shift_y == 0 else -shift_y),
        slice(None if shift_z == 0 else -shift_z),
    )
    return arr[src_shifted] - arr[src_base]


def _sum_and_count(values: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(values)
    return float(np.sum(values[valid], dtype=np.float64)), int(np.count_nonzero(valid))


def compute_velocity_sf_reduction_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    shift_x: int,
    shift_y: int,
    shift_z: int,
    sf_type: tuple[str, ...],
    boundary=None,
) -> dict[str, float | int]:
    """Return local sum/count contributions for one 3D velocity displacement."""
    du = shifted_difference_3d(u, shift_x, shift_y, shift_z, boundary=boundary)
    dv = shifted_difference_3d(v, shift_x, shift_y, shift_z, boundary=boundary)
    dw = shifted_difference_3d(w, shift_x, shift_y, shift_z, boundary=boundary)

    lx = float(shift_x)
    ly = float(shift_y)
    lz = float(shift_z)
    radius = np.sqrt(lx * lx + ly * ly + lz * lz)

    if radius == 0.0:
        raise ValueError("The zero displacement vector is not valid for SF evaluation.")

    d_long = (lx * du + ly * dv + lz * dw) / radius
    d_perp_sq = du * du + dv * dv + dw * dw - d_long * d_long
    d_perp_sq = np.maximum(d_perp_sq, 0.0)

    reductions: dict[str, float | int] = {}
    if "LL" in sf_type:
        reductions["SF_LL_sum"], reductions["SF_LL_count"] = _sum_and_count(d_long**2)
    if "TT" in sf_type:
        reductions["SF_TT_sum"], reductions["SF_TT_count"] = _sum_and_count(d_perp_sq)
    if "LLL" in sf_type:
        reductions["SF_LLL_sum"], reductions["SF_LLL_count"] = _sum_and_count(d_long**3)
    if "LTT" in sf_type:
        reductions["SF_LTT_sum"], reductions["SF_LTT_count"] = _sum_and_count(
            d_long * d_perp_sq
        )
    return reductions


def finalize_structure_function_reduction(
    reductions: dict[str, float | int], sf_type: tuple[str, ...]
) -> dict[str, float]:
    """Convert local sum/count reductions into mean structure functions."""
    output = {}
    for name in sf_type:
        count = int(reductions[f"SF_{name}_count"])
        if count == 0:
            output[f"SF_{name}"] = float("nan")
            continue
        output[f"SF_{name}"] = float(reductions[f"SF_{name}_sum"]) / count
    return output


def compute_scalar_sf_reduction_3d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    scalar: np.ndarray,
    shift_x: int,
    shift_y: int,
    shift_z: int,
    sf_type: tuple[str, ...],
    boundary=None,
) -> dict[str, float | int]:
    """Return local sum/count contributions for one 3D scalar displacement."""
    du = shifted_difference_3d(u, shift_x, shift_y, shift_z, boundary=boundary)
    dv = shifted_difference_3d(v, shift_x, shift_y, shift_z, boundary=boundary)
    dw = shifted_difference_3d(w, shift_x, shift_y, shift_z, boundary=boundary)
    ds = shifted_difference_3d(scalar, shift_x, shift_y, shift_z, boundary=boundary)

    lx = float(shift_x)
    ly = float(shift_y)
    lz = float(shift_z)
    radius = np.sqrt(lx * lx + ly * ly + lz * lz)

    if radius == 0.0:
        raise ValueError("The zero displacement vector is not valid for SF evaluation.")

    d_long = (lx * du + ly * dv + lz * dw) / radius
    reductions: dict[str, float | int] = {}
    if "SS" in sf_type:
        reductions["SF_SS_sum"], reductions["SF_SS_count"] = _sum_and_count(ds**2)
    if "LSS" in sf_type:
        reductions["SF_LSS_sum"], reductions["SF_LSS_count"] = _sum_and_count(
            d_long * ds**2
        )
    return reductions


def finalize_velocity_sf_reduction(
    reductions: dict[str, float | int], sf_type: tuple[str, ...]
) -> dict[str, float]:
    """Backward-compatible alias for velocity reduction finalization."""
    return finalize_structure_function_reduction(reductions, sf_type)


__all__ = (
    "compute_scalar_sf_reduction_3d",
    "compute_velocity_sf_reduction_3d",
    "finalize_structure_function_reduction",
    "finalize_velocity_sf_reduction",
    "shifted_difference_3d",
)
