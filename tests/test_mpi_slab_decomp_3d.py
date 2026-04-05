import numpy as np

from fluidsf.calculate_advection_3d import calculate_advection_3d
from fluidsf.calculate_structure_function_3d import calculate_structure_function_3d
from fluidsf.mpi.slab_decomp_3d import (
    calculate_advection_3d_public_x_slab_mpi,
    compute_advective_sf_direction_3d_periodic_z_slab_mpi,
    compute_advective_sf_direction_3d_public_x_slab_mpi,
    compute_directional_sf_3d_public_x_slab_mpi,
    compute_slab_bounds_1d,
    compute_scalar_sf_reduction_3d_periodic_z_slab_mpi,
    generate_sf_grid_3d_periodic_z_slab_mpi,
    compute_velocity_sf_reduction_3d_periodic_z_slab_mpi,
    exchange_periodic_halo_z,
)


class _FakeCommSingleRank:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Sendrecv(self, sendbuf, dest, sendtag, recvbuf, source, recvtag):
        recvbuf[...] = sendbuf

    def allgather(self, value):
        return [value]

    def allreduce(self, value):
        return value


def test_compute_slab_bounds_1d_even_and_uneven():
    assert compute_slab_bounds_1d(8, 4, 0) == (0, 2)
    assert compute_slab_bounds_1d(8, 4, 3) == (6, 8)
    assert compute_slab_bounds_1d(10, 4, 0) == (0, 3)
    assert compute_slab_bounds_1d(10, 4, 1) == (3, 6)
    assert compute_slab_bounds_1d(10, 4, 2) == (6, 8)
    assert compute_slab_bounds_1d(10, 4, 3) == (8, 10)


def test_exchange_periodic_halo_z_single_rank_wraps_core():
    comm = _FakeCommSingleRank()
    arr = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)

    extended = exchange_periodic_halo_z(arr, 2, comm)

    np.testing.assert_allclose(extended[:, :, :2], arr[:, :, -2:])
    np.testing.assert_allclose(extended[:, :, 2:6], arr)
    np.testing.assert_allclose(extended[:, :, 6:], arr[:, :, :2])


def test_compute_velocity_sf_reduction_3d_periodic_z_slab_mpi_single_rank():
    comm = _FakeCommSingleRank()
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

    sf = compute_velocity_sf_reduction_3d_periodic_z_slab_mpi(
        u,
        v,
        w,
        0,
        0,
        1,
        ("LL", "TT", "LLL", "LTT"),
        comm=comm,
    )

    assert np.isclose(sf["SF_LL"], 3.0)
    assert np.isclose(sf["SF_TT"], 0.0)
    assert np.isclose(sf["SF_LLL"], -6.0)
    assert np.isclose(sf["SF_LTT"], 0.0)


def test_generate_sf_grid_3d_periodic_z_slab_mpi_single_rank():
    comm = _FakeCommSingleRank()
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

    sf = generate_sf_grid_3d_periodic_z_slab_mpi(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=("LL", "TT", "LLL", "LTT"),
        comm=comm,
    )

    assert sf["SF_LL_grid"].shape == (2, 2, 2)
    assert np.isclose(sf["SF_LL_grid"][1, 0, 0], 3.0)
    assert np.isclose(sf["SF_LL_grid"][0, 1, 0], 3.0)
    assert np.isclose(sf["SF_LL_grid"][0, 0, 1], 3.0)
    assert np.isclose(sf["SF_LLL_grid"][1, 0, 0], -6.0)


def test_compute_scalar_sf_reduction_3d_periodic_z_slab_mpi_single_rank():
    comm = _FakeCommSingleRank()
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]
    scalar = u + 2.0 * v

    sf = compute_scalar_sf_reduction_3d_periodic_z_slab_mpi(
        u,
        v,
        w,
        scalar,
        1,
        0,
        0,
        ("SS", "LSS"),
        comm=comm,
    )

    assert np.isclose(sf["SF_SS"], 3.0)
    assert np.isclose(sf["SF_LSS"], -6.0)


def test_calculate_advection_3d_public_x_slab_mpi_matches_serial_convention():
    comm = _FakeCommSingleRank()
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u = 2.0 * xx + 7.0 * zz
    v = 3.0 * yy + 11.0 * xx
    w = 5.0 * zz + 13.0 * yy

    u_adv, v_adv, w_adv = calculate_advection_3d_public_x_slab_mpi(
        u, v, w, x, y, z, comm=comm
    )
    serial_u_adv, serial_v_adv, serial_w_adv = calculate_advection_3d(u, v, w, x, y, z)

    np.testing.assert_allclose(u_adv, serial_u_adv)
    np.testing.assert_allclose(v_adv, serial_v_adv)
    np.testing.assert_allclose(w_adv, serial_w_adv)

    scalar = 2.0 * xx + 3.0 * yy + 5.0 * zz + 17.0 * xx * yy
    scalar_adv = calculate_advection_3d_public_x_slab_mpi(
        u, v, w, x, y, z, scalar_local=scalar, comm=comm
    )
    serial_scalar_adv = calculate_advection_3d(u, v, w, x, y, z, scalar=scalar)
    np.testing.assert_allclose(scalar_adv, serial_scalar_adv)


def test_compute_directional_sf_3d_public_x_slab_mpi_matches_serial_convention():
    comm = _FakeCommSingleRank()
    x = np.arange(5, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(3, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u = 2.0 * xx + 7.0 * zz
    v = 3.0 * yy + 11.0 * xx
    w = 5.0 * zz + 13.0 * yy

    for direction, serial_key in (("x", "SF_LL_x"), ("y", "SF_LL_y"), ("z", "SF_LL_z")):
        helper = compute_directional_sf_3d_public_x_slab_mpi(
            u,
            v,
            w,
            direction=direction,
            shift=1,
            sf_type=("LL", "TT"),
            boundary=None,
            comm=comm,
        )
        serial = calculate_structure_function_3d(
            u,
            v,
            w,
            None,
            None,
            None,
            1,
            1,
            1,
            ("LL", "TT"),
            boundary=None,
        )
        np.testing.assert_allclose(helper["SF_LL"], serial[serial_key])
        np.testing.assert_allclose(helper["SF_TT"], serial[f"SF_TT_{direction}"])

def test_compute_directional_sf_3d_public_x_slab_mpi_periodic_matches_serial_convention():
    comm = _FakeCommSingleRank()
    x = np.arange(5, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u = 2.0 * xx + 7.0 * zz
    v = 3.0 * yy + 11.0 * xx
    w = 5.0 * zz + 13.0 * yy

    for direction, serial_key in (("x", "SF_LL_x"), ("y", "SF_LL_y"), ("z", "SF_LL_z")):
        helper = compute_directional_sf_3d_public_x_slab_mpi(
            u,
            v,
            w,
            direction=direction,
            shift=1,
            sf_type=("LL", "TT"),
            boundary="periodic-all",
            comm=comm,
        )
        serial = calculate_structure_function_3d(
            u,
            v,
            w,
            None,
            None,
            None,
            1,
            1,
            1,
            ("LL", "TT"),
            boundary="periodic-all",
        )
        np.testing.assert_allclose(helper["SF_LL"], serial[serial_key])
        np.testing.assert_allclose(helper["SF_TT"], serial[f"SF_TT_{direction}"])


def test_advective_sf_helper_zero_count_returns_nan():
    comm = _FakeCommSingleRank()
    arr = np.empty((0, 1, 1))

    public = compute_advective_sf_direction_3d_public_x_slab_mpi(
        arr,
        arr,
        arr,
        direction="x",
        shift=1,
        adv_x_local=arr,
        adv_y_local=arr,
        adv_z_local=arr,
        sf_type=("ASF_V",),
        comm=comm,
    )
    z_slab_arr = np.empty((1, 1, 0))
    z_slab = compute_advective_sf_direction_3d_periodic_z_slab_mpi(
        z_slab_arr,
        z_slab_arr,
        z_slab_arr,
        direction="x",
        shift=1,
        adv_x_local=z_slab_arr,
        adv_y_local=z_slab_arr,
        adv_z_local=z_slab_arr,
        sf_type=("ASF_V",),
        comm=comm,
    )

    assert np.isnan(public["SF_advection_velocity"])
    assert np.isnan(z_slab["SF_advection_velocity"])
