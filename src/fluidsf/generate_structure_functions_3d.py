import warnings

import numpy as np

from .bin_data import bin_data
from .mpi.slab_decomp_3d import (
    calculate_advection_3d_public_x_slab_mpi,
    compute_directional_sf_3d_public_x_slab_mpi,
    compute_slab_bounds_1d,
)


def _requested_structure_functions(sf_type):
    """Return the exact 3D SF requests made by the caller."""
    requested = set(sf_type)
    return {
        "ASF_V": "ASF_V" in requested,
        "ASF_S": "ASF_S" in requested,
        "LL": "LL" in requested,
        "TT": "TT" in requested,
        "SS": "SS" in requested,
        "LLL": "LLL" in requested,
        "LTT": "LTT" in requested,
        "LSS": "LSS" in requested,
        "LLLL": "LLLL" in requested,
    }


def _boundary_is_periodic_all(boundary):
    if boundary is None:
        return False
    if isinstance(boundary, str):
        return "periodic-all" in boundary
    return any("periodic-all" in entry for entry in boundary)


def _boundary_entries(boundary):
    if boundary is None:
        return ()
    if isinstance(boundary, str):
        return (boundary,)
    return tuple(boundary)


def _axis_has_periodic_extent(boundary, axis_name):
    entries = _boundary_entries(boundary)
    return ("periodic-all" in entries) or (f"periodic-{axis_name}" in entries)


def _axis_uses_periodic_shifts(boundary, axis_name):
    if boundary is None:
        return False
    if isinstance(boundary, str):
        return "periodic-all" in boundary
    entries = tuple(boundary)
    return ("periodic-all" in entries) or (f"periodic-{axis_name}" in entries)


def _normalize_public_3d_inputs(u, v, w, x, y, z, scalar=None, backend="serial"):
    """Return public-layout arrays shaped ``(len(x), len(y), len(z))``.

    The 3D public API historically mixed two conventions:
    - the documented/public layout ``(x, y, z)``
    - a legacy internal layout ``(z, y, x)`` used by older examples

    Normalize the legacy replicated layout up front so both the serial and MPI
    paths evaluate the same physical problem.
    """
    if u.shape != v.shape or u.shape != w.shape:
        raise ValueError("u, v, and w must have identical shapes.")
    if u.ndim != 3:
        raise ValueError("u, v, and w must be 3D arrays.")
    if scalar is not None and scalar.shape != u.shape:
        raise ValueError("scalar must match u, v, and w.")

    public_shape = (len(x), len(y), len(z))
    legacy_shape = (len(z), len(y), len(x))

    if u.shape == public_shape:
        normalized_scalar = None if scalar is None else np.ascontiguousarray(scalar)
        return (
            np.ascontiguousarray(u),
            np.ascontiguousarray(v),
            np.ascontiguousarray(w),
            normalized_scalar,
        )

    if u.shape == legacy_shape:
        if backend == "mpi" and u.shape[0] != len(z):
            raise ValueError(
                "Legacy 3D inputs must be replicated arrays shaped (len(z), len(y), len(x))."
            )
        normalized_scalar = None
        if scalar is not None:
            normalized_scalar = np.ascontiguousarray(np.transpose(scalar, (2, 1, 0)))
        return (
            np.ascontiguousarray(np.transpose(u, (2, 1, 0))),
            np.ascontiguousarray(np.transpose(v, (2, 1, 0))),
            np.ascontiguousarray(np.transpose(w, (2, 1, 0))),
            normalized_scalar,
        )

    if backend == "mpi" and u.shape[1:] == (len(y), len(z)):
        normalized_scalar = None if scalar is None else np.ascontiguousarray(scalar)
        return (
            np.ascontiguousarray(u),
            np.ascontiguousarray(v),
            np.ascontiguousarray(w),
            normalized_scalar,
        )

    raise ValueError(
        "3D inputs must use the public layout (len(x), len(y), len(z)) or the "
        "legacy replicated layout (len(z), len(y), len(x))."
    )


def _shift_public_3d_array(arr, shift, axis, periodic):
    """Shift a public-layout ``(x, y, z)`` array forward along one axis."""
    if shift == 0:
        return np.ascontiguousarray(arr)
    if periodic:
        return np.roll(arr, shift=-shift, axis=axis)

    shifted = np.full(arr.shape, np.nan, dtype=np.result_type(arr, np.float64))
    dst = [slice(None), slice(None), slice(None)]
    src = [slice(None), slice(None), slice(None)]
    dst[axis] = slice(None, -shift)
    src[axis] = slice(shift, None)
    shifted[tuple(dst)] = arr[tuple(src)]
    return shifted


def _calculate_advection_3d_public(u, v, w, x, y, z, scalar=None):
    """Distributed-API-consistent 3D advection for public-layout arrays."""
    dx = np.abs(x[0] - x[1])
    dy = np.abs(y[0] - y[1])
    dz = np.abs(z[0] - z[1])

    if scalar is not None:
        dsdx, dsdy, dsdz = np.gradient(scalar, dx, dy, dz, axis=(0, 1, 2))
        return u * dsdx + v * dsdy + w * dsdz

    dudx, dudy, dudz = np.gradient(u, dx, dy, dz, axis=(0, 1, 2))
    dvdx, dvdy, dvdz = np.gradient(v, dx, dy, dz, axis=(0, 1, 2))
    dwdx, dwdy, dwdz = np.gradient(w, dx, dy, dz, axis=(0, 1, 2))

    u_advection = u * dudx + v * dudy + w * dudz
    v_advection = u * dvdx + v * dvdy + w * dvdz
    w_advection = u * dwdx + v * dwdy + w * dwdz
    return u_advection, v_advection, w_advection


def _compute_directional_sf_3d_public(
    u,
    v,
    w,
    *,
    direction,
    shift,
    sf_type,
    boundary,
    scalar=None,
    adv_x=None,
    adv_y=None,
    adv_z=None,
    adv_scalar=None,
):
    """Compute one axis-aligned 3D structure-function slice on public arrays."""
    requested = _requested_structure_functions(sf_type)
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis = axis_map[direction]
    periodic = _axis_uses_periodic_shifts(boundary, direction)

    u_shift = _shift_public_3d_array(u, shift, axis, periodic)
    v_shift = _shift_public_3d_array(v, shift, axis, periodic)
    w_shift = _shift_public_3d_array(w, shift, axis, periodic)
    du = u_shift - u
    dv = v_shift - v
    dw = w_shift - w

    if direction == "x":
        d_long = du
        trans_a = dv
        trans_b = dw
    elif direction == "y":
        d_long = dv
        trans_a = du
        trans_b = dw
    else:
        d_long = dw
        trans_a = du
        trans_b = dv

    ds = None
    if scalar is not None and any(requested[name] for name in ("ASF_S", "SS", "LSS")):
        scalar_shift = _shift_public_3d_array(scalar, shift, axis, periodic)
        ds = scalar_shift - scalar

    output = {}
    if requested["ASF_V"]:
        adv_x_shift = _shift_public_3d_array(adv_x, shift, axis, periodic)
        adv_y_shift = _shift_public_3d_array(adv_y, shift, axis, periodic)
        adv_z_shift = _shift_public_3d_array(adv_z, shift, axis, periodic)
        output[f"SF_advection_velocity_{direction}"] = np.nanmean(
            (adv_x_shift - adv_x) * du
            + (adv_y_shift - adv_y) * dv
            + (adv_z_shift - adv_z) * dw
        )
    if requested["ASF_S"]:
        adv_scalar_shift = _shift_public_3d_array(adv_scalar, shift, axis, periodic)
        output[f"SF_advection_scalar_{direction}"] = np.nanmean(
            (adv_scalar_shift - adv_scalar) * ds
        )
    if requested["LL"]:
        output[f"SF_LL_{direction}"] = np.nanmean(d_long**2)
    if requested["TT"]:
        output[f"SF_TT_{direction}"] = np.nanmean(trans_a**2 + trans_b**2)
    if requested["SS"]:
        output[f"SF_SS_{direction}"] = np.nanmean(ds**2)
    if requested["LLL"]:
        output[f"SF_LLL_{direction}"] = np.nanmean(d_long**3)
    if requested["LTT"]:
        output[f"SF_LTT_{direction}"] = np.nanmean(d_long * (trans_a**2 + trans_b**2))
    if requested["LSS"]:
        output[f"SF_LSS_{direction}"] = np.nanmean(d_long * ds**2)
    return output


def _generate_structure_functions_3d_mpi_backend(
    u,
    v,
    w,
    x,
    y,
    z,
    sf_type,
    scalar,
    boundary,
    nbins,
    px,
    comm,
):
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    requested = _requested_structure_functions(sf_type)
    if requested["ASF_S"] and scalar is None:
        raise ValueError("scalar is required for ASF_S in the MPI backend.")
    if (requested["SS"] or requested["LSS"]) and scalar is None:
        raise ValueError("scalar is required for SS or LSS in the MPI backend.")
    supported_boundary_entries = {"periodic-all", "periodic-x", "periodic-y", "periodic-z"}
    unsupported_boundary_entries = set(_boundary_entries(boundary)) - supported_boundary_entries
    if unsupported_boundary_entries:
        raise ValueError(
            "The MPI backend supports only boundary=None and the periodic boundary "
            "entries: periodic-x, periodic-y, periodic-z, periodic-all."
        )
    u, v, w, scalar = _normalize_public_3d_inputs(
        u, v, w, x, y, z, scalar=scalar, backend="mpi"
    )

    size = comm.Get_size()
    total_x = comm.allreduce(u.shape[0])
    expects_public_layout = u.shape[1] == len(y) and u.shape[2] == len(z)

    if expects_public_layout and total_x == len(x):
        # Distributed input: each rank owns a slab in the public x dimension.
        u_public_local = np.ascontiguousarray(u)
        v_public_local = np.ascontiguousarray(v)
        w_public_local = np.ascontiguousarray(w)
        scalar_public_local = None
        if scalar is not None:
            scalar_public_local = np.ascontiguousarray(scalar)
    elif u.shape == (len(x), len(y), len(z)):
        # Replicated input: extract the public-layout slab owned by this rank.
        start, stop = compute_slab_bounds_1d(len(x), size, comm.Get_rank())
        u_public_local = np.ascontiguousarray(u[start:stop, :, :])
        v_public_local = np.ascontiguousarray(v[start:stop, :, :])
        w_public_local = np.ascontiguousarray(w[start:stop, :, :])
        scalar_public_local = None
        if scalar is not None:
            scalar_public_local = np.ascontiguousarray(scalar[start:stop, :, :])
    else:
        raise ValueError(
            "The MPI backend expects either full arrays shaped (len(x), len(y), len(z)) "
            "on every rank or distributed x-slabs shaped (local_x, len(y), len(z)) "
            "whose local_x sizes sum to len(x)."
        )

    sep_x = range(1, int(len(x) / 2)) if _axis_has_periodic_extent(boundary, "x") else range(1, int(len(x) - 1))
    sep_y = range(1, int(len(y) / 2)) if _axis_has_periodic_extent(boundary, "y") else range(1, int(len(y) - 1))
    sep_z = range(1, int(len(z) / 2)) if _axis_has_periodic_extent(boundary, "z") else range(1, int(len(z) - 1))

    output = {
        "x-diffs": np.asarray(x[: len(sep_x) + 1], dtype=np.float64) - float(x[0]),
        "y-diffs": np.asarray(y[: len(sep_y) + 1], dtype=np.float64) - float(y[0]),
        "z-diffs": np.asarray(z[: len(sep_z) + 1], dtype=np.float64) - float(z[0]),
    }

    direct_sf_type = tuple(
        name
        for name in ("ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS")
        if requested[name]
    )
    if direct_sf_type:
        adv_velocity = None
        adv_scalar = None
        if requested["ASF_V"]:
            adv_velocity = calculate_advection_3d_public_x_slab_mpi(
                u_public_local,
                v_public_local,
                w_public_local,
                x,
                y,
                z,
                comm=comm,
            )
        if requested["ASF_S"]:
            adv_scalar = calculate_advection_3d_public_x_slab_mpi(
                u_public_local,
                v_public_local,
                w_public_local,
                x,
                y,
                z,
                scalar_local=scalar_public_local,
                comm=comm,
            )

        axis_map = (
            ("x", sep_x),
            ("y", sep_y),
            ("z", sep_z),
        )
        key_map = {
            "ASF_V": "SF_advection_velocity",
            "ASF_S": "SF_advection_scalar",
            "LL": "SF_LL",
            "TT": "SF_TT",
            "SS": "SF_SS",
            "LLL": "SF_LLL",
            "LTT": "SF_LTT",
            "LSS": "SF_LSS",
        }

        for name in direct_sf_type:
            for axis_name, sep in axis_map:
                output[f"{key_map[name]}_{axis_name}"] = np.zeros(len(sep) + 1)

        for axis_name, sep in axis_map:
            for shift in sep:
                boundary_mode = (
                    "periodic-all" if _axis_uses_periodic_shifts(boundary, axis_name) else None
                )
                reduced = compute_directional_sf_3d_public_x_slab_mpi(
                    u_public_local,
                    v_public_local,
                    w_public_local,
                    direction=axis_name,
                    shift=shift,
                    sf_type=direct_sf_type,
                    boundary=boundary_mode,
                    scalar_local=scalar_public_local,
                    adv_x_local=None if adv_velocity is None else adv_velocity[0],
                    adv_y_local=None if adv_velocity is None else adv_velocity[1],
                    adv_z_local=None if adv_velocity is None else adv_velocity[2],
                    adv_scalar_local=adv_scalar,
                    comm=comm,
                )

                for name in direct_sf_type:
                    output[f"{key_map[name]}_{axis_name}"][shift] = reduced[key_map[name]]

    if nbins is not None and output["x-diffs"] is not None:
        if requested["ASF_V"]:
            xd_bin, output["SF_advection_velocity_x"] = bin_data(
                output["x-diffs"], output["SF_advection_velocity_x"], nbins
            )
            yd_bin, output["SF_advection_velocity_y"] = bin_data(
                output["y-diffs"], output["SF_advection_velocity_y"], nbins
            )
            zd_bin, output["SF_advection_velocity_z"] = bin_data(
                output["z-diffs"], output["SF_advection_velocity_z"], nbins
            )
        if requested["ASF_S"]:
            xd_bin, output["SF_advection_scalar_x"] = bin_data(
                output["x-diffs"], output["SF_advection_scalar_x"], nbins
            )
            yd_bin, output["SF_advection_scalar_y"] = bin_data(
                output["y-diffs"], output["SF_advection_scalar_y"], nbins
            )
            zd_bin, output["SF_advection_scalar_z"] = bin_data(
                output["z-diffs"], output["SF_advection_scalar_z"], nbins
            )
        if requested["LL"]:
            xd_bin, output["SF_LL_x"] = bin_data(output["x-diffs"], output["SF_LL_x"], nbins)
            yd_bin, output["SF_LL_y"] = bin_data(output["y-diffs"], output["SF_LL_y"], nbins)
            zd_bin, output["SF_LL_z"] = bin_data(output["z-diffs"], output["SF_LL_z"], nbins)
        if requested["TT"]:
            xd_bin, output["SF_TT_x"] = bin_data(output["x-diffs"], output["SF_TT_x"], nbins)
            yd_bin, output["SF_TT_y"] = bin_data(output["y-diffs"], output["SF_TT_y"], nbins)
            zd_bin, output["SF_TT_z"] = bin_data(output["z-diffs"], output["SF_TT_z"], nbins)
        if requested["SS"]:
            xd_bin, output["SF_SS_x"] = bin_data(output["x-diffs"], output["SF_SS_x"], nbins)
            yd_bin, output["SF_SS_y"] = bin_data(output["y-diffs"], output["SF_SS_y"], nbins)
            zd_bin, output["SF_SS_z"] = bin_data(output["z-diffs"], output["SF_SS_z"], nbins)
        if requested["LLL"]:
            xd_bin, output["SF_LLL_x"] = bin_data(output["x-diffs"], output["SF_LLL_x"], nbins)
            yd_bin, output["SF_LLL_y"] = bin_data(output["y-diffs"], output["SF_LLL_y"], nbins)
            zd_bin, output["SF_LLL_z"] = bin_data(output["z-diffs"], output["SF_LLL_z"], nbins)
        if requested["LTT"]:
            xd_bin, output["SF_LTT_x"] = bin_data(output["x-diffs"], output["SF_LTT_x"], nbins)
            yd_bin, output["SF_LTT_y"] = bin_data(output["y-diffs"], output["SF_LTT_y"], nbins)
            zd_bin, output["SF_LTT_z"] = bin_data(output["z-diffs"], output["SF_LTT_z"], nbins)
        if requested["LSS"]:
            xd_bin, output["SF_LSS_x"] = bin_data(output["x-diffs"], output["SF_LSS_x"], nbins)
            yd_bin, output["SF_LSS_y"] = bin_data(output["y-diffs"], output["SF_LSS_y"], nbins)
            zd_bin, output["SF_LSS_z"] = bin_data(output["z-diffs"], output["SF_LSS_z"], nbins)
        output["x-diffs"] = xd_bin
        output["y-diffs"] = yd_bin
        output["z-diffs"] = zd_bin
    return output


def generate_structure_functions_3d(  # noqa: C901, D417
    u,
    v,
    w,
    x,
    y,
    z,
    sf_type=["ASF_V"],  # noqa: B006
    scalar=None,
    boundary="periodic-all",
    nbins=None,
    backend="serial",
    px=1,
    comm=None,
):
    """
    Full method for generating structure functions for uniform and even 3D data,
    including advective structure functions. Supports velocity-based and
    scalar-based structure functions. Defaults to calculating the
    velocity-based advective structure functions for the x, y, and z directions.

    Parameters
    ----------
        u: ndarray
            3D array of u velocity components.
        v: ndarray
            3D array of v velocity components.
        w: ndarray
            3D array of w velocity components.
        x: ndarray
            1D array of x-coordinates.
        y: ndarray
            1D array of y-coordinates.
        z: ndarray
            1D array of z-coordinates.
        sf_type: list
            List of structure function types to calculate.
            Accepted list entries must be one or more of the following strings:
            "ASF_V, "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS".
            Defaults to ["ASF_V"].
        scalar: ndarray, optional
            3D array of scalar values. Defaults to None.
        boundary: str, optional
            Boundary condition of the data. Accepted strings are "periodic-x",
            "periodic-y", "periodic-z", and "periodic-all". Defaults to "periodic-all".
        nbins: int, optional
            Number of bins in the structure function. Defaults to None, i.e. does
            not bin the data.
        backend: str, optional
            Execution backend. ``"serial"`` evaluates the public ``(x, y, z)``
            layout directly while still accepting the legacy replicated
            ``(z, y, x)`` layout. ``"mpi"`` enables the distributed public-layout
            backend.
        px: int, optional
            Reserved for MPI backend compatibility. The current distributed
            backend ignores this value.
        comm: mpi4py.MPI.Comm, optional
            Optional communicator for the MPI backend. Defaults to
            ``MPI.COMM_WORLD`` when ``backend="mpi"``.

    Returns
    -------
        dict:
            Dictionary containing the requested structure functions and separation
            distances for the x, y, and z directions.
            The returned dictionary may contain the following keys, with some keys
            removed if the structure function is not calculated:

                **SF_advection_velocity_x**: The advective velocity structure function
                in the x direction.

                **SF_advection_velocity_y**: The advective velocity structure function
                in the y direction.

                **SF_advection_velocity_z**: The advective velocity structure function
                in the z direction.

                **SF_advection_scalar_x**: The advective scalar structure function
                in the x direction.

                **SF_advection_scalar_y**: The advective scalar structure function
                in the y direction.

                **SF_advection_scalar_z**: The advective scalar structure function
                in the z direction.

                **SF_LL_x**: The second-order longitudinal velocity structure function
                in the x direction.

                **SF_LL_y**: The second-order longitudinal velocity structure function
                in the y direction.

                **SF_LL_z**: The second-order longitudinal velocity structure function
                in the z direction.

                **SF_TT_x**: The second-order transverse velocity structure function
                in the x direction.

                **SF_TT_y**: The second-order transverse velocity structure function
                in the y direction.

                **SF_TT_z**: The second-order transverse velocity structure function
                in the z direction.

                **SF_SS_x**: The second-order scalar structure function in the x
                direction.

                **SF_SS_y**: The second-order scalar structure function in the y
                direction.

                **SF_SS_z**: The second-order scalar structure function in the z
                direction.

                **SF_LLL_x**: The third-order longitudinal velocity structure function
                in the x direction.

                **SF_LLL_y**: The third-order longitudinal velocity structure function
                in the y direction.

                **SF_LLL_z**: The third-order longitudinal velocity structure function
                in the z direction.

                **SF_LTT_x**: The third-order longitudinal-transverse-transverse
                velocity structure function in the x direction.

                **SF_LTT_y**: The third-order longitudinal-transverse-transverse
                velocity structure function in the y direction.

                **SF_LTT_z**: The third-order longitudinal-transverse-transverse
                velocity structure function in the z direction.

                **SF_LSS_x**: The third-order longitudinal-scalar-scalar velocity
                structure function in the x direction.

                **SF_LSS_y**: The third-order longitudinal-scalar-scalar velocity
                structure function in the y direction.

                **SF_LSS_z**: The third-order longitudinal-scalar-scalar velocity
                structure function in the z direction.

                **x-diffs**: The separation distances in the x direction.

                **y-diffs**: The separation distances in the y direction.

                **z-diffs**: The separation distances in the z direction.

    """
    if backend == "mpi":
        return _generate_structure_functions_3d_mpi_backend(
            u,
            v,
            w,
            x,
            y,
            z,
            sf_type,
            scalar,
            boundary,
            nbins,
            px,
            comm,
        )
    if backend != "serial":
        raise ValueError("backend must be either 'serial' or 'mpi'.")

    u, v, w, scalar = _normalize_public_3d_inputs(
        u, v, w, x, y, z, scalar=scalar, backend="serial"
    )

    # Initialize variables as NoneType
    SF_adv_x = None
    SF_adv_y = None
    SF_adv_z = None
    SF_x_scalar = None
    SF_y_scalar = None
    SF_z_scalar = None
    adv_x = None
    adv_y = None
    adv_z = None
    adv_scalar = None
    SF_x_LL = None
    SF_y_LL = None
    SF_z_LL = None
    SF_x_TT = None
    SF_y_TT = None
    SF_z_TT = None
    SF_x_SS = None
    SF_y_SS = None
    SF_z_SS = None
    SF_x_LLL = None
    SF_y_LLL = None
    SF_z_LLL = None
    SF_x_LTT = None
    SF_y_LTT = None
    SF_z_LTT = None
    SF_x_LSS = None
    SF_y_LSS = None
    SF_z_LSS = None

    # Define a list of separation distances to iterate over.
    # Periodic is half the length since the calculation will wrap the data.
    sep_x = range(1, int(len(x) - 1))
    sep_y = range(1, int(len(y) - 1))
    sep_z = range(1, int(len(z) - 1))

    if boundary is not None:
        if "periodic-all" in boundary:
            sep_x = range(1, int(len(x) / 2))
            sep_y = range(1, int(len(y) / 2))
            sep_z = range(1, int(len(z) / 2))
        if "periodic-x" in boundary:
            sep_x = range(1, int(len(x) / 2))
        if "periodic-y" in boundary:
            sep_y = range(1, int(len(y) / 2))
        if "periodic-z" in boundary:
            sep_z = range(1, int(len(z) / 2))

    # Initialize the separation distance arrays
    xd = np.zeros(len(sep_x) + 1)
    yd = np.zeros(len(sep_y) + 1)
    zd = np.zeros(len(sep_z) + 1)

    # Initialize the structure function arrays
    requested = _requested_structure_functions(sf_type)

    if requested["ASF_V"]:
        SF_adv_x = np.zeros(len(sep_x) + 1)
        SF_adv_y = np.zeros(len(sep_y) + 1)
        SF_adv_z = np.zeros(len(sep_z) + 1)
        adv_x, adv_y, adv_z = _calculate_advection_3d_public(u, v, w, x, y, z)
    if requested["ASF_S"]:
        SF_x_scalar = np.zeros(len(sep_x) + 1)
        SF_y_scalar = np.zeros(len(sep_y) + 1)
        SF_z_scalar = np.zeros(len(sep_z) + 1)
        adv_scalar = _calculate_advection_3d_public(u, v, w, x, y, z, scalar)
    if requested["LL"]:
        SF_x_LL = np.zeros(len(sep_x) + 1)
        SF_y_LL = np.zeros(len(sep_y) + 1)
        SF_z_LL = np.zeros(len(sep_z) + 1)
    if requested["TT"]:
        SF_x_TT = np.zeros(len(sep_x) + 1)
        SF_y_TT = np.zeros(len(sep_y) + 1)
        SF_z_TT = np.zeros(len(sep_z) + 1)
    if requested["SS"]:
        SF_x_SS = np.zeros(len(sep_x) + 1)
        SF_y_SS = np.zeros(len(sep_y) + 1)
        SF_z_SS = np.zeros(len(sep_z) + 1)
    if requested["LLL"]:
        SF_x_LLL = np.zeros(len(sep_x) + 1)
        SF_y_LLL = np.zeros(len(sep_y) + 1)
        SF_z_LLL = np.zeros(len(sep_z) + 1)
    if requested["LTT"]:
        SF_x_LTT = np.zeros(len(sep_x) + 1)
        SF_y_LTT = np.zeros(len(sep_y) + 1)
        SF_z_LTT = np.zeros(len(sep_z) + 1)
    if requested["LSS"]:
        SF_x_LSS = np.zeros(len(sep_x) + 1)
        SF_y_LSS = np.zeros(len(sep_y) + 1)
        SF_z_LSS = np.zeros(len(sep_z) + 1)
    if requested["LLLL"]:
        warnings.warn(
            "Structure functions of order 4 or higher are not implemented in "
            "generate_structure_functions_3d and will be ignored.",
            stacklevel=2,
        )

    # Iterate over separations in x, y, and z
    for x_shift in sep_x:
        SF_dicts = _compute_directional_sf_3d_public(
            u,
            v,
            w,
            direction="x",
            shift=x_shift,
            sf_type=sf_type,
            boundary=boundary,
            scalar=scalar,
            adv_x=adv_x,
            adv_y=adv_y,
            adv_z=adv_z,
            adv_scalar=adv_scalar,
        )

        if requested["ASF_V"]:
            SF_adv_x[x_shift] = SF_dicts["SF_advection_velocity_x"]
        if requested["ASF_S"]:
            SF_x_scalar[x_shift] = SF_dicts["SF_advection_scalar_x"]
        if requested["LL"]:
            SF_x_LL[x_shift] = SF_dicts["SF_LL_x"]
        if requested["TT"]:
            SF_x_TT[x_shift] = SF_dicts["SF_TT_x"]
        if requested["SS"]:
            SF_x_SS[x_shift] = SF_dicts["SF_SS_x"]
        if requested["LLL"]:
            SF_x_LLL[x_shift] = SF_dicts["SF_LLL_x"]
        if requested["LTT"]:
            SF_x_LTT[x_shift] = SF_dicts["SF_LTT_x"]
        if requested["LSS"]:
            SF_x_LSS[x_shift] = SF_dicts["SF_LSS_x"]

        xd[x_shift] = float(np.abs(x[x_shift] - x[0]))

    for y_shift in sep_y:
        SF_dicts = _compute_directional_sf_3d_public(
            u,
            v,
            w,
            direction="y",
            shift=y_shift,
            sf_type=sf_type,
            boundary=boundary,
            scalar=scalar,
            adv_x=adv_x,
            adv_y=adv_y,
            adv_z=adv_z,
            adv_scalar=adv_scalar,
        )

        if requested["ASF_V"]:
            SF_adv_y[y_shift] = SF_dicts["SF_advection_velocity_y"]
        if requested["ASF_S"]:
            SF_y_scalar[y_shift] = SF_dicts["SF_advection_scalar_y"]
        if requested["LL"]:
            SF_y_LL[y_shift] = SF_dicts["SF_LL_y"]
        if requested["TT"]:
            SF_y_TT[y_shift] = SF_dicts["SF_TT_y"]
        if requested["SS"]:
            SF_y_SS[y_shift] = SF_dicts["SF_SS_y"]
        if requested["LLL"]:
            SF_y_LLL[y_shift] = SF_dicts["SF_LLL_y"]
        if requested["LTT"]:
            SF_y_LTT[y_shift] = SF_dicts["SF_LTT_y"]
        if requested["LSS"]:
            SF_y_LSS[y_shift] = SF_dicts["SF_LSS_y"]

        yd[y_shift] = float(np.abs(y[y_shift] - y[0]))

    for z_shift in sep_z:
        SF_dicts = _compute_directional_sf_3d_public(
            u,
            v,
            w,
            direction="z",
            shift=z_shift,
            sf_type=sf_type,
            boundary=boundary,
            scalar=scalar,
            adv_x=adv_x,
            adv_y=adv_y,
            adv_z=adv_z,
            adv_scalar=adv_scalar,
        )

        if requested["ASF_V"]:
            SF_adv_z[z_shift] = SF_dicts["SF_advection_velocity_z"]
        if requested["ASF_S"]:
            SF_z_scalar[z_shift] = SF_dicts["SF_advection_scalar_z"]
        if requested["LL"]:
            SF_z_LL[z_shift] = SF_dicts["SF_LL_z"]
        if requested["TT"]:
            SF_z_TT[z_shift] = SF_dicts["SF_TT_z"]
        if requested["SS"]:
            SF_z_SS[z_shift] = SF_dicts["SF_SS_z"]
        if requested["LLL"]:
            SF_z_LLL[z_shift] = SF_dicts["SF_LLL_z"]
        if requested["LTT"]:
            SF_z_LTT[z_shift] = SF_dicts["SF_LTT_z"]
        if requested["LSS"]:
            SF_z_LSS[z_shift] = SF_dicts["SF_LSS_z"]

        zd[z_shift] = float(np.abs(z[z_shift] - z[0]))

    if nbins is not None:
        if requested["ASF_V"]:
            xd_bin, SF_adv_x = bin_data(xd, SF_adv_x, nbins)
            yd_bin, SF_adv_y = bin_data(yd, SF_adv_y, nbins)
            zd_bin, SF_adv_z = bin_data(zd, SF_adv_z, nbins)
        if requested["ASF_S"]:
            xd_bin, SF_x_scalar = bin_data(xd, SF_x_scalar, nbins)
            yd_bin, SF_y_scalar = bin_data(yd, SF_y_scalar, nbins)
            zd_bin, SF_z_scalar = bin_data(zd, SF_z_scalar, nbins)
        if requested["LL"]:
            xd_bin, SF_x_LL = bin_data(xd, SF_x_LL, nbins)
            yd_bin, SF_y_LL = bin_data(yd, SF_y_LL, nbins)
            zd_bin, SF_z_LL = bin_data(zd, SF_z_LL, nbins)
        if requested["TT"]:
            xd_bin, SF_x_TT = bin_data(xd, SF_x_TT, nbins)
            yd_bin, SF_y_TT = bin_data(yd, SF_y_TT, nbins)
            zd_bin, SF_z_TT = bin_data(zd, SF_z_TT, nbins)
        if requested["SS"]:
            xd_bin, SF_x_SS = bin_data(xd, SF_x_SS, nbins)
            yd_bin, SF_y_SS = bin_data(yd, SF_y_SS, nbins)
            zd_bin, SF_z_SS = bin_data(zd, SF_z_SS, nbins)
        if requested["LLL"]:
            xd_bin, SF_x_LLL = bin_data(xd, SF_x_LLL, nbins)
            yd_bin, SF_y_LLL = bin_data(yd, SF_y_LLL, nbins)
            zd_bin, SF_z_LLL = bin_data(zd, SF_z_LLL, nbins)
        if requested["LTT"]:
            xd_bin, SF_x_LTT = bin_data(xd, SF_x_LTT, nbins)
            yd_bin, SF_y_LTT = bin_data(yd, SF_y_LTT, nbins)
            zd_bin, SF_z_LTT = bin_data(zd, SF_z_LTT, nbins)
        if requested["LSS"]:
            xd_bin, SF_x_LSS = bin_data(xd, SF_x_LSS, nbins)
            yd_bin, SF_y_LSS = bin_data(yd, SF_y_LSS, nbins)
            zd_bin, SF_z_LSS = bin_data(zd, SF_z_LSS, nbins)
        xd = xd_bin
        yd = yd_bin
        zd = zd_bin

    data = {
        key: value
        for key, value in {
            "SF_advection_velocity_x": SF_adv_x,
            "SF_advection_velocity_y": SF_adv_y,
            "SF_advection_velocity_z": SF_adv_z,
            "SF_advection_scalar_x": SF_x_scalar,
            "SF_advection_scalar_y": SF_y_scalar,
            "SF_advection_scalar_z": SF_z_scalar,
            "SF_LL_x": SF_x_LL,
            "SF_LL_y": SF_y_LL,
            "SF_LL_z": SF_z_LL,
            "SF_TT_x": SF_x_TT,
            "SF_TT_y": SF_y_TT,
            "SF_TT_z": SF_z_TT,
            "SF_SS_x": SF_x_SS,
            "SF_SS_y": SF_y_SS,
            "SF_SS_z": SF_z_SS,
            "SF_LLL_x": SF_x_LLL,
            "SF_LLL_y": SF_y_LLL,
            "SF_LLL_z": SF_z_LLL,
            "SF_LTT_x": SF_x_LTT,
            "SF_LTT_y": SF_y_LTT,
            "SF_LTT_z": SF_z_LTT,
            "SF_LSS_x": SF_x_LSS,
            "SF_LSS_y": SF_y_LSS,
            "SF_LSS_z": SF_z_LSS,
            "x-diffs": xd,
            "y-diffs": yd,
            "z-diffs": zd,
        }.items()
        if value is not None
    }
    return data
