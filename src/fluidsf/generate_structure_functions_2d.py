import warnings

import numpy as np

from .bin_data import bin_data
from .calculate_advection_2d import calculate_advection_2d
from .calculate_separation_distances import calculate_separation_distances
from .calculate_structure_function_2d import (
    calculate_structure_function_2d,
)
from .mpi.slab_decomp_2d import (
    calculate_advection_2d_public_x_slab_mpi,
    compute_directional_sf_2d_public_x_slab_mpi,
    compute_slab_bounds_1d,
)
from .shift_array_1d import shift_array_1d


def _generate_structure_functions_2d_mpi_backend_legacy(
    u,
    v,
    x,
    y,
    sf_type,
    scalar,
    dx,
    dy,
    boundary,
    grid_type,
    nbins,
    comm,
):
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    if boundary == "periodic-all":
        sep_x = range(1, int(len(x) / 2))
        sep_y = range(1, int(len(y) / 2))
    elif boundary == "periodic-x":
        sep_x = range(1, int(len(x) / 2))
        sep_y = range(1, int(len(y) - 1))
    elif boundary == "periodic-y":
        sep_x = range(1, int(len(x) - 1))
        sep_y = range(1, int(len(y) / 2))
    else:
        sep_x = range(1, int(len(x) - 1))
        sep_y = range(1, int(len(y) - 1))

    rank = comm.Get_rank()
    size = comm.Get_size()

    xd = np.zeros(len(sep_x) + 1)
    yd = np.zeros(len(sep_y) + 1)
    sf_arrays = {}

    if any("ASF_V" in t for t in sf_type):
        sf_arrays["SF_advection_velocity_x"] = np.zeros(len(sep_x) + 1)
        sf_arrays["SF_advection_velocity_y"] = np.zeros(len(sep_y) + 1)
        adv_x, adv_y = calculate_advection_2d(u, v, x, y, dx, dy, grid_type)
    else:
        adv_x = adv_y = None

    if any("ASF_S" in t for t in sf_type):
        sf_arrays["SF_advection_scalar_x"] = np.zeros(len(sep_x) + 1)
        sf_arrays["SF_advection_scalar_y"] = np.zeros(len(sep_y) + 1)
        adv_scalar = calculate_advection_2d(u, v, x, y, dx, dy, grid_type, scalar)
    else:
        adv_scalar = None

    if any("LL" in t for t in sf_type):
        sf_arrays["SF_LL_x"] = np.zeros(len(sep_x) + 1)
        sf_arrays["SF_LL_y"] = np.zeros(len(sep_y) + 1)
    if any("TT" in t for t in sf_type):
        sf_arrays["SF_TT_x"] = np.zeros(len(sep_x) + 1)
        sf_arrays["SF_TT_y"] = np.zeros(len(sep_y) + 1)
    if any("SS" in t for t in sf_type):
        sf_arrays["SF_SS_x"] = np.zeros(len(sep_x) + 1)
        sf_arrays["SF_SS_y"] = np.zeros(len(sep_y) + 1)
    if any("LLL" in t for t in sf_type):
        sf_arrays["SF_LLL_x"] = np.zeros(len(sep_x) + 1)
        sf_arrays["SF_LLL_y"] = np.zeros(len(sep_y) + 1)
    if any("LTT" in t for t in sf_type):
        sf_arrays["SF_LTT_x"] = np.zeros(len(sep_x) + 1)
        sf_arrays["SF_LTT_y"] = np.zeros(len(sep_y) + 1)
    if any("LSS" in t for t in sf_type):
        sf_arrays["SF_LSS_x"] = np.zeros(len(sep_x) + 1)
        sf_arrays["SF_LSS_y"] = np.zeros(len(sep_y) + 1)

    for local_index, x_shift in enumerate(sep_x, start=1):
        if (local_index - 1) % size != rank:
            continue
        y_shift = 1
        if boundary == "periodic-all" or boundary == "periodic-x":
            xroll = shift_array_1d(x, shift_by=x_shift, boundary="Periodic")
        else:
            xroll = shift_array_1d(x, shift_by=x_shift, boundary=None)

        sf_dict = calculate_structure_function_2d(
            u,
            v,
            adv_x,
            adv_y,
            x_shift,
            y_shift,
            sf_type,
            scalar,
            adv_scalar,
            boundary,
        )

        for key in ("SF_advection_velocity_x", "SF_LL_x", "SF_TT_x", "SF_SS_x", "SF_LLL_x", "SF_LTT_x", "SF_advection_scalar_x", "SF_LSS_x"):
            if key in sf_arrays:
                sf_arrays[key][x_shift] = sf_dict[key]

        xd[x_shift], _ = calculate_separation_distances(
            x[0], y[0], xroll[0], y[0], grid_type
        )

    for local_index, y_shift in enumerate(sep_y, start=1):
        if (local_index - 1) % size != rank:
            continue
        x_shift = 1
        if boundary == "periodic-all" or boundary == "periodic-y":
            yroll = shift_array_1d(y, shift_by=y_shift, boundary="Periodic")
        else:
            yroll = shift_array_1d(y, shift_by=y_shift, boundary=None)

        sf_dict = calculate_structure_function_2d(
            u,
            v,
            adv_x,
            adv_y,
            x_shift,
            y_shift,
            sf_type,
            scalar,
            adv_scalar,
            boundary,
        )

        for key in ("SF_advection_velocity_y", "SF_LL_y", "SF_TT_y", "SF_SS_y", "SF_LLL_y", "SF_LTT_y", "SF_advection_scalar_y", "SF_LSS_y"):
            if key in sf_arrays:
                sf_arrays[key][y_shift] = sf_dict[key]

        _, yd[y_shift] = calculate_separation_distances(
            x[0], y[0], x[0], yroll[0], grid_type
        )

    xd = comm.allreduce(xd)
    yd = comm.allreduce(yd)
    for key in list(sf_arrays):
        sf_arrays[key] = comm.allreduce(sf_arrays[key])

    if nbins is not None:
        if any("ASF_V" in t for t in sf_type):
            xd_bin, sf_arrays["SF_advection_velocity_x"] = bin_data(
                xd, sf_arrays["SF_advection_velocity_x"], nbins
            )
            yd_bin, sf_arrays["SF_advection_velocity_y"] = bin_data(
                yd, sf_arrays["SF_advection_velocity_y"], nbins
            )
        if any("ASF_S" in t for t in sf_type):
            xd_bin, sf_arrays["SF_advection_scalar_x"] = bin_data(
                xd, sf_arrays["SF_advection_scalar_x"], nbins
            )
            yd_bin, sf_arrays["SF_advection_scalar_y"] = bin_data(
                yd, sf_arrays["SF_advection_scalar_y"], nbins
            )
        if any("LL" in t for t in sf_type):
            xd_bin, sf_arrays["SF_LL_x"] = bin_data(xd, sf_arrays["SF_LL_x"], nbins)
            yd_bin, sf_arrays["SF_LL_y"] = bin_data(yd, sf_arrays["SF_LL_y"], nbins)
        if any("TT" in t for t in sf_type):
            xd_bin, sf_arrays["SF_TT_x"] = bin_data(xd, sf_arrays["SF_TT_x"], nbins)
            yd_bin, sf_arrays["SF_TT_y"] = bin_data(yd, sf_arrays["SF_TT_y"], nbins)
        if any("SS" in t for t in sf_type):
            xd_bin, sf_arrays["SF_SS_x"] = bin_data(xd, sf_arrays["SF_SS_x"], nbins)
            yd_bin, sf_arrays["SF_SS_y"] = bin_data(yd, sf_arrays["SF_SS_y"], nbins)
        if any("LLL" in t for t in sf_type):
            xd_bin, sf_arrays["SF_LLL_x"] = bin_data(xd, sf_arrays["SF_LLL_x"], nbins)
            yd_bin, sf_arrays["SF_LLL_y"] = bin_data(yd, sf_arrays["SF_LLL_y"], nbins)
        if any("LTT" in t for t in sf_type):
            xd_bin, sf_arrays["SF_LTT_x"] = bin_data(xd, sf_arrays["SF_LTT_x"], nbins)
            yd_bin, sf_arrays["SF_LTT_y"] = bin_data(yd, sf_arrays["SF_LTT_y"], nbins)
        if any("LSS" in t for t in sf_type):
            xd_bin, sf_arrays["SF_LSS_x"] = bin_data(xd, sf_arrays["SF_LSS_x"], nbins)
            yd_bin, sf_arrays["SF_LSS_y"] = bin_data(yd, sf_arrays["SF_LSS_y"], nbins)
        xd = xd_bin
        yd = yd_bin

    return {
        key: value
        for key, value in {
            **sf_arrays,
            "x-diffs": xd,
            "y-diffs": yd,
        }.items()
        if value is not None
    }


def _generate_structure_functions_2d_mpi_backend(
    u,
    v,
    x,
    y,
    sf_type,
    scalar,
    dx,
    dy,
    boundary,
    grid_type,
    nbins,
    comm,
):
    if comm is None:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    if grid_type != "uniform":
        return _generate_structure_functions_2d_mpi_backend_legacy(
            u, v, x, y, sf_type, scalar, dx, dy, boundary, grid_type, nbins, comm
        )

    if u.shape != v.shape:
        raise ValueError("u and v must have identical shapes.")

    size = comm.Get_size()
    total_x = comm.allreduce(u.shape[1])
    expects_public_layout = u.shape[0] == len(y)

    if expects_public_layout and total_x == len(x):
        u_local = np.ascontiguousarray(u)
        v_local = np.ascontiguousarray(v)
        scalar_local = None if scalar is None else np.ascontiguousarray(scalar)
    elif u.shape == (len(y), len(x)):
        start, stop = compute_slab_bounds_1d(len(x), size, comm.Get_rank())
        u_local = np.ascontiguousarray(u[:, start:stop])
        v_local = np.ascontiguousarray(v[:, start:stop])
        scalar_local = None if scalar is None else np.ascontiguousarray(scalar[:, start:stop])
    else:
        return _generate_structure_functions_2d_mpi_backend_legacy(
            u, v, x, y, sf_type, scalar, dx, dy, boundary, grid_type, nbins, comm
        )

    if boundary == "periodic-all":
        sep_x = range(1, int(len(x) / 2))
        sep_y = range(1, int(len(y) / 2))
    elif boundary == "periodic-x":
        sep_x = range(1, int(len(x) / 2))
        sep_y = range(1, int(len(y) - 1))
    elif boundary == "periodic-y":
        sep_x = range(1, int(len(x) - 1))
        sep_y = range(1, int(len(y) / 2))
    else:
        sep_x = range(1, int(len(x) - 1))
        sep_y = range(1, int(len(y) - 1))

    xd = np.zeros(len(sep_x) + 1)
    yd = np.zeros(len(sep_y) + 1)
    output = {"x-diffs": xd, "y-diffs": yd}

    requested = {
        "ASF_V": any("ASF_V" in t for t in sf_type),
        "ASF_S": any("ASF_S" in t for t in sf_type),
        "LL": any("LL" in t for t in sf_type),
        "TT": any("TT" in t for t in sf_type),
        "SS": any("SS" in t for t in sf_type),
        "LLL": any("LLL" in t for t in sf_type),
        "LTT": any("LTT" in t for t in sf_type),
        "LSS": any("LSS" in t for t in sf_type),
    }
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

    for name, enabled in requested.items():
        if not enabled:
            continue
        output[f"{key_map[name]}_x"] = np.zeros(len(sep_x) + 1)
        output[f"{key_map[name]}_y"] = np.zeros(len(sep_y) + 1)

    u_int_local = np.ascontiguousarray(u_local.T)
    v_int_local = np.ascontiguousarray(v_local.T)
    scalar_int_local = None if scalar_local is None else np.ascontiguousarray(scalar_local.T)

    adv_x = adv_y = adv_scalar = None
    if requested["ASF_V"]:
        adv_x, adv_y = calculate_advection_2d_public_x_slab_mpi(
            u_int_local, v_int_local, x, y, layout="internal", comm=comm
        )
    if requested["ASF_S"]:
        adv_scalar = calculate_advection_2d_public_x_slab_mpi(
            u_int_local,
            v_int_local,
            x,
            y,
            scalar_local=scalar_int_local,
            layout="internal",
            comm=comm,
        )

    requested_names = tuple(name for name, enabled in requested.items() if enabled)
    for x_shift in sep_x:
        sf_dict = compute_directional_sf_2d_public_x_slab_mpi(
            u_int_local,
            v_int_local,
            direction="x",
            shift=x_shift,
            sf_type=requested_names,
            boundary=boundary,
            scalar_local=scalar_int_local,
            adv_x_local=adv_x,
            adv_y_local=adv_y,
            adv_scalar_local=adv_scalar,
            layout="internal",
            comm=comm,
        )
        for name in requested_names:
            output[f"{key_map[name]}_x"][x_shift] = sf_dict[key_map[name]]
        xroll = shift_array_1d(
            x,
            shift_by=x_shift,
            boundary="Periodic" if boundary in {"periodic-x", "periodic-all"} else None,
        )
        xd[x_shift], _ = calculate_separation_distances(x[0], y[0], xroll[0], y[0], grid_type)

    for y_shift in sep_y:
        sf_dict = compute_directional_sf_2d_public_x_slab_mpi(
            u_int_local,
            v_int_local,
            direction="y",
            shift=y_shift,
            sf_type=requested_names,
            boundary=boundary,
            scalar_local=scalar_int_local,
            adv_x_local=adv_x,
            adv_y_local=adv_y,
            adv_scalar_local=adv_scalar,
            layout="internal",
            comm=comm,
        )
        for name in requested_names:
            output[f"{key_map[name]}_y"][y_shift] = sf_dict[key_map[name]]
        yroll = shift_array_1d(
            y,
            shift_by=y_shift,
            boundary="Periodic" if boundary in {"periodic-y", "periodic-all"} else None,
        )
        _, yd[y_shift] = calculate_separation_distances(x[0], y[0], x[0], yroll[0], grid_type)

    if nbins is not None:
        if requested["ASF_V"]:
            xd_bin, output["SF_advection_velocity_x"] = bin_data(
                output["x-diffs"], output["SF_advection_velocity_x"], nbins
            )
            yd_bin, output["SF_advection_velocity_y"] = bin_data(
                output["y-diffs"], output["SF_advection_velocity_y"], nbins
            )
        if requested["ASF_S"]:
            xd_bin, output["SF_advection_scalar_x"] = bin_data(
                output["x-diffs"], output["SF_advection_scalar_x"], nbins
            )
            yd_bin, output["SF_advection_scalar_y"] = bin_data(
                output["y-diffs"], output["SF_advection_scalar_y"], nbins
            )
        if requested["LL"]:
            xd_bin, output["SF_LL_x"] = bin_data(output["x-diffs"], output["SF_LL_x"], nbins)
            yd_bin, output["SF_LL_y"] = bin_data(output["y-diffs"], output["SF_LL_y"], nbins)
        if requested["TT"]:
            xd_bin, output["SF_TT_x"] = bin_data(output["x-diffs"], output["SF_TT_x"], nbins)
            yd_bin, output["SF_TT_y"] = bin_data(output["y-diffs"], output["SF_TT_y"], nbins)
        if requested["SS"]:
            xd_bin, output["SF_SS_x"] = bin_data(output["x-diffs"], output["SF_SS_x"], nbins)
            yd_bin, output["SF_SS_y"] = bin_data(output["y-diffs"], output["SF_SS_y"], nbins)
        if requested["LLL"]:
            xd_bin, output["SF_LLL_x"] = bin_data(output["x-diffs"], output["SF_LLL_x"], nbins)
            yd_bin, output["SF_LLL_y"] = bin_data(output["y-diffs"], output["SF_LLL_y"], nbins)
        if requested["LTT"]:
            xd_bin, output["SF_LTT_x"] = bin_data(output["x-diffs"], output["SF_LTT_x"], nbins)
            yd_bin, output["SF_LTT_y"] = bin_data(output["y-diffs"], output["SF_LTT_y"], nbins)
        if requested["LSS"]:
            xd_bin, output["SF_LSS_x"] = bin_data(output["x-diffs"], output["SF_LSS_x"], nbins)
            yd_bin, output["SF_LSS_y"] = bin_data(output["y-diffs"], output["SF_LSS_y"], nbins)
        output["x-diffs"] = xd_bin
        output["y-diffs"] = yd_bin

    return output


def generate_structure_functions_2d(  # noqa: C901, D417
    u,
    v,
    x,
    y,
    sf_type=["ASF_V"],  # noqa: B006
    scalar=None,
    dx=None,
    dy=None,
    boundary="periodic-all",
    grid_type="uniform",
    nbins=None,
    backend="serial",
    comm=None,
):
    """
    Full method for generating structure functions for 2D data, including advective
    structure functions. Supports velocity-based and scalar-based structure
    functions. Defaults to calculating the velocity-based advective structure functions
    for the x and y directions.

    Parameters
    ----------
        u: ndarray
            2D array of u velocity components.
        v: ndarray
            2D array of v velocity components.
        x: ndarray
            1D array of x-coordinates.
        y: ndarray
            1D array of y-coordinates.
        sf_type: list
            List of structure function types to calculate.
            Accepted list entries must be one or more of the following strings:
            "ASF_V, "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS".
            Defaults to ["ASF_V"].
        scalar: ndarray, optional
            2D array of scalar values. Defaults to None.
        dx: float, optional
            Grid spacing in the x-direction. Defaults to None.
        dy: float, optional
            Grid spacing in the y-direction. Defaults to None.
        boundary: str, optional
            Boundary condition of the data. Accepted strings are "periodic-x",
            "periodic-y", and "periodic-all". Defaults to "periodic-all".
        grid_type:str, optional
            Type of grid, either "uniform" or "latlon". Defaults to "uniform".
        nbins: int, optional
            Number of bins for binning the data. Defaults to None, i.e. does not bin
            data.
        backend: str, optional
            Execution backend. ``"serial"`` uses the legacy single-process
            implementation. ``"mpi"`` distributes separation calculations across MPI
            ranks while expecting the full 2D arrays on every rank.
        comm: optional
            MPI communicator used when ``backend="mpi"``.

    Returns
    -------
        dict:
            Dictionary containing the requested structure functions and separation
            distances for the x- and y-direction.
            The returned dictionary may contain the following keys, with some keys
            removed if the structure function is not calculated:

                **SF_advection_velocity_x**: The advective velocity structure function
                in the x direction.

                **SF_advection_velocity_y**: The advective velocity structure function
                in the y direction.

                **SF_advection_scalar_x**: The advective scalar structure function
                in the x direction.

                **SF_advection_scalar_y**: The advective scalar structure function
                in the y direction.

                **SF_LL_x**: The second-order longitudinal velocity structure function
                in the x direction.

                **SF_LL_y**: The second-order longitudinal velocity structure function
                in the y direction.

                **SF_TT_x**: The second-order transverse velocity structure function
                in the x direction.

                **SF_TT_y**: The second-order transverse velocity structure function
                in the y direction.

                **SF_SS_x**: The second-order scalar structure function in the x
                direction.

                **SF_SS_y**: The second-order scalar structure function in the y
                direction.

                **SF_LLL_x**: The third-order longitudinal velocity structure function
                in the x direction.

                **SF_LLL_y**: The third-order longitudinal velocity structure function
                in the y direction.

                **SF_LTT_x**: The third-order longitudinal-transverse-transverse
                velocity structure function in the x direction.

                **SF_LTT_y**: The third-order longitudinal-transverse-transverse
                velocity structure function in the y direction.

                **SF_LSS_x**: The third-order longitudinal-scalar-scalar structure
                function in the x direction.

                **SF_LSS_y**: The third-order longitudinal-scalar-scalar structure
                function in the y direction.

                **x-diffs**: The separation distances in the x direction.

                **y-diffs**: The separation distances in the y direction.

    """
    # Error handling

    if not isinstance(sf_type, list):
        raise ValueError("sf_type must be a list of strings.")

    if len(sf_type) == 0:
        raise ValueError(
            "sf_type cannot be an empty list. All elements in sf_type must be strings. "
            "Accepted strings are: ASF_V, ASF_S, LL, TT, SS, LLL, LTT, LSS."
        )

    if not all(isinstance(t, str) for t in sf_type):
        raise ValueError(
            "All elements in sf_type must be strings. Accepted strings are: "
            "ASF_V, ASF_S, LL, TT, SS, LLL, LTT, LSS."
        )

    if boundary not in ["periodic-all", "periodic-x", "periodic-y", None]:
        raise ValueError(
            "Boundary must be 'periodic-all', 'periodic-x', 'periodic-y', or None."
        )
    if grid_type not in ["uniform", "latlon"]:
        raise ValueError("Grid type must be 'uniform' or 'latlon'.")

    if grid_type == "latlon" and (
        isinstance(dx, int | float | None) or isinstance(dy, int | float | None)
    ):
        raise ValueError(
            "If grid_type is 'latlon', dx and dy must be provided as arrays."
        )

    if scalar is not None and (
        ("SS" not in sf_type) and ("LSS" not in sf_type) and ("ASF_S" not in sf_type)
    ):
        raise ValueError(
            "If scalar is provided, you must include 'SS', 'LSS' or 'ASF_S' "
            "in SF_type."
        )
    if scalar is None and (
        ("SS" in sf_type) or ("LSS" in sf_type) or ("ASF_S" in sf_type)
    ):
        raise ValueError(
            "If you include 'SS', 'LSS' or 'ASF_S' in SF_type, you must provide "
            "a scalar array."
        )
    if backend not in ["serial", "mpi"]:
        raise ValueError("backend must be 'serial' or 'mpi'.")
    if backend == "mpi":
        return _generate_structure_functions_2d_mpi_backend(
            u,
            v,
            x,
            y,
            sf_type,
            scalar,
            dx,
            dy,
            boundary,
            grid_type,
            nbins,
            comm,
        )

    # Initialize variables as NoneType
    SF_adv_x = None
    SF_adv_y = None
    SF_x_scalar = None
    SF_y_scalar = None
    adv_x = None
    adv_y = None
    adv_scalar = None
    SF_x_LL = None
    SF_y_LL = None
    SF_x_TT = None
    SF_y_TT = None
    SF_x_SS = None
    SF_y_SS = None
    SF_x_LLL = None
    SF_y_LLL = None
    SF_x_LTT = None
    SF_y_LTT = None
    SF_x_LSS = None
    SF_y_LSS = None

    # Define a list of separation distances to iterate over.
    # Periodic is half the length since the calculation will wrap the data.
    if boundary == "periodic-all":
        sep_x = range(1, int(len(x) / 2))
        sep_y = range(1, int(len(y) / 2))
    elif boundary == "periodic-x":
        sep_x = range(1, int(len(x) / 2))
        sep_y = range(1, int(len(y) - 1))
    elif boundary == "periodic-y":
        sep_x = range(1, int(len(x) - 1))
        sep_y = range(1, int(len(y) / 2))
    elif boundary is None:
        sep_x = range(1, int(len(x) - 1))
        sep_y = range(1, int(len(y) - 1))

    # Initialize the separation distance arrays
    xd = np.zeros(len(sep_x) + 1)
    yd = np.zeros(len(sep_y) + 1)

    # Calculate advection if requested
    if any("ASF_V" in t for t in sf_type):
        SF_adv_x = np.zeros(len(sep_x) + 1)
        SF_adv_y = np.zeros(len(sep_y) + 1)
        adv_x, adv_y = calculate_advection_2d(u, v, x, y, dx, dy, grid_type)
    if any("ASF_S" in t for t in sf_type):
        SF_x_scalar = np.zeros(len(sep_x) + 1)
        SF_y_scalar = np.zeros(len(sep_y) + 1)
        adv_scalar = calculate_advection_2d(u, v, x, y, dx, dy, grid_type, scalar)
    if any("LL" in t for t in sf_type):
        SF_x_LL = np.zeros(len(sep_x) + 1)
        SF_y_LL = np.zeros(len(sep_y) + 1)
    if any("TT" in t for t in sf_type):
        SF_x_TT = np.zeros(len(sep_x) + 1)
        SF_y_TT = np.zeros(len(sep_y) + 1)
    if any("SS" in t for t in sf_type):
        SF_x_SS = np.zeros(len(sep_x) + 1)
        SF_y_SS = np.zeros(len(sep_y) + 1)
    if any("LLL" in t for t in sf_type):
        SF_x_LLL = np.zeros(len(sep_x) + 1)
        SF_y_LLL = np.zeros(len(sep_y) + 1)
    if any("LTT" in t for t in sf_type):
        SF_x_LTT = np.zeros(len(sep_x) + 1)
        SF_y_LTT = np.zeros(len(sep_y) + 1)
    if any("LSS" in t for t in sf_type):
        SF_x_LSS = np.zeros(len(sep_x) + 1)
        SF_y_LSS = np.zeros(len(sep_y) + 1)
    if any("LLLL" in t for t in sf_type):
        warnings.warn(
            "Structure functions of order 4 or higher require manual coding. "
            "LL and LLL will still be computed.",
            stacklevel=2,
        )

    # Iterate over separations in x and y
    for x_shift in sep_x:
        y_shift = 1
        if boundary == "periodic-all" or boundary == "periodic-x":
            xroll = shift_array_1d(x, shift_by=x_shift, boundary="Periodic")
        else:
            xroll = shift_array_1d(x, shift_by=x_shift, boundary=None)

        SF_dicts = calculate_structure_function_2d(
            u,
            v,
            adv_x,
            adv_y,
            x_shift,
            y_shift,
            sf_type,
            scalar,
            adv_scalar,
            boundary,
        )

        if any("ASF_V" in t for t in sf_type):
            SF_adv_x[x_shift] = SF_dicts["SF_advection_velocity_x"]
        if any("LL" in t for t in sf_type):
            SF_x_LL[x_shift] = SF_dicts["SF_LL_x"]
        if any("TT" in t for t in sf_type):
            SF_x_TT[x_shift] = SF_dicts["SF_TT_x"]
        if any("SS" in t for t in sf_type):
            SF_x_SS[x_shift] = SF_dicts["SF_SS_x"]
        if any("LLL" in t for t in sf_type):
            SF_x_LLL[x_shift] = SF_dicts["SF_LLL_x"]
        if any("LTT" in t for t in sf_type):
            SF_x_LTT[x_shift] = SF_dicts["SF_LTT_x"]
        if any("ASF_S" in t for t in sf_type):
            SF_x_scalar[x_shift] = SF_dicts["SF_advection_scalar_x"]
        if any("LSS" in t for t in sf_type):
            SF_x_LSS[x_shift] = SF_dicts["SF_LSS_x"]

        # Calculate separation distances in x
        xd[x_shift], tmp = calculate_separation_distances(
            x[0], y[0], xroll[0], y[0], grid_type
        )

    for y_shift in sep_y:
        x_shift = 1
        if boundary == "periodic-all" or boundary == "periodic-y":
            yroll = shift_array_1d(y, shift_by=y_shift, boundary="Periodic")
        else:
            yroll = shift_array_1d(y, shift_by=y_shift, boundary=None)

        SF_dicts = calculate_structure_function_2d(
            u,
            v,
            adv_x,
            adv_y,
            x_shift,
            y_shift,
            sf_type,
            scalar,
            adv_scalar,
            boundary,
        )

        if any("ASF_V" in t for t in sf_type):
            SF_adv_y[y_shift] = SF_dicts["SF_advection_velocity_y"]
        if any("LL" in t for t in sf_type):
            SF_y_LL[y_shift] = SF_dicts["SF_LL_y"]
        if any("TT" in t for t in sf_type):
            SF_y_TT[y_shift] = SF_dicts["SF_TT_y"]
        if any("SS" in t for t in sf_type):
            SF_y_SS[y_shift] = SF_dicts["SF_SS_y"]
        if any("LLL" in t for t in sf_type):
            SF_y_LLL[y_shift] = SF_dicts["SF_LLL_y"]
        if any("LTT" in t for t in sf_type):
            SF_y_LTT[y_shift] = SF_dicts["SF_LTT_y"]
        if any("ASF_S" in t for t in sf_type):
            SF_y_scalar[y_shift] = SF_dicts["SF_advection_scalar_y"]
        if any("LSS" in t for t in sf_type):
            SF_y_LSS[y_shift] = SF_dicts["SF_LSS_y"]

        # Calculate separation distances in y
        tmp, yd[y_shift] = calculate_separation_distances(
            x[0], y[0], x[0], yroll[0], grid_type
        )

    # Bin the data if requested
    if nbins is not None:
        if any("ASF_V" in t for t in sf_type):
            xd_bin, SF_adv_x = bin_data(xd, SF_adv_x, nbins)
            yd_bin, SF_adv_y = bin_data(yd, SF_adv_y, nbins)
        if any("ASF_S" in t for t in sf_type):
            xd_bin, SF_x_scalar = bin_data(xd, SF_x_scalar, nbins)
            yd_bin, SF_y_scalar = bin_data(yd, SF_y_scalar, nbins)
        if any("LL" in t for t in sf_type):
            xd_bin, SF_x_LL = bin_data(xd, SF_x_LL, nbins)
            yd_bin, SF_y_LL = bin_data(yd, SF_y_LL, nbins)
        if any("TT" in t for t in sf_type):
            xd_bin, SF_x_TT = bin_data(xd, SF_x_TT, nbins)
            yd_bin, SF_y_TT = bin_data(yd, SF_y_TT, nbins)
        if any("SS" in t for t in sf_type):
            xd_bin, SF_x_SS = bin_data(xd, SF_x_SS, nbins)
            yd_bin, SF_y_SS = bin_data(yd, SF_y_SS, nbins)
        if any("LLL" in t for t in sf_type):
            xd_bin, SF_x_LLL = bin_data(xd, SF_x_LLL, nbins)
            yd_bin, SF_y_LLL = bin_data(yd, SF_y_LLL, nbins)
        if any("LTT" in t for t in sf_type):
            xd_bin, SF_x_LTT = bin_data(xd, SF_x_LTT, nbins)
            yd_bin, SF_y_LTT = bin_data(yd, SF_y_LTT, nbins)
        if any("LSS" in t for t in sf_type):
            xd_bin, SF_x_LSS = bin_data(xd, SF_x_LSS, nbins)
            yd_bin, SF_y_LSS = bin_data(yd, SF_y_LSS, nbins)
        xd = xd_bin
        yd = yd_bin

    data = {
        key: value
        for key, value in {
            "SF_advection_velocity_x": SF_adv_x,
            "SF_advection_velocity_y": SF_adv_y,
            "SF_advection_scalar_x": SF_x_scalar,
            "SF_advection_scalar_y": SF_y_scalar,
            "SF_LL_x": SF_x_LL,
            "SF_LL_y": SF_y_LL,
            "SF_TT_x": SF_x_TT,
            "SF_TT_y": SF_y_TT,
            "SF_SS_x": SF_x_SS,
            "SF_SS_y": SF_y_SS,
            "SF_LLL_x": SF_x_LLL,
            "SF_LLL_y": SF_y_LLL,
            "SF_LTT_x": SF_x_LTT,
            "SF_LTT_y": SF_y_LTT,
            "SF_LSS_x": SF_x_LSS,
            "SF_LSS_y": SF_y_LSS,
            "x-diffs": xd,
            "y-diffs": yd,
        }.items()
        if value is not None
    }

    return data
