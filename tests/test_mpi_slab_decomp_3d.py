import numpy as np

from fluidsf.mpi.slab_decomp_3d import (
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
