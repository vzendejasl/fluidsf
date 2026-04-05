import numpy as np

from fluidsf.mpi.generate_sf_3d_mpi import (
    _compute_velocity_structure_functions,
    generate_sf_grid_3d_mpi,
)


class _FakeComm:
    def __init__(self):
        self._rank = 0
        self._size = 1

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def gather(self, value, root=0):
        assert root == 0
        return [value]


def test_compute_velocity_structure_functions_linear_field():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

    sf = _compute_velocity_structure_functions(
        u, v, w, 1, 0, 0, ("LL", "TT", "LLL", "LTT"), boundary=None
    )
    assert np.isclose(sf["SF_LL"], 1.0)
    assert np.isclose(sf["SF_TT"], 0.0)
    assert np.isclose(sf["SF_LLL"], 1.0)
    assert np.isclose(sf["SF_LTT"], 0.0)


def test_generate_sf_grid_3d_mpi_single_rank():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

    sf = generate_sf_grid_3d_mpi(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL", "TT"],
        px=1,
        boundary=None,
        comm=_FakeComm(),
    )

    assert sf["SF_LL_grid"].shape == (2, 2, 2)
    assert sf["SF_TT_grid"].shape == (2, 2, 2)
    np.testing.assert_allclose(sf["x-diffs"], [0.0, 1.0])
    np.testing.assert_allclose(sf["y-diffs"], [0.0, 1.0])
    np.testing.assert_allclose(sf["z-diffs"], [0.0, 1.0])

    assert np.isclose(sf["SF_LL_grid"][1, 0, 0], 1.0)
    assert np.isclose(sf["SF_LL_grid"][0, 1, 0], 1.0)
    assert np.isclose(sf["SF_LL_grid"][0, 0, 1], 1.0)
    assert np.isclose(sf["SF_TT_grid"][1, 0, 0], 0.0)
    assert np.isclose(sf["SF_TT_grid"][0, 1, 0], 0.0)
    assert np.isclose(sf["SF_TT_grid"][0, 0, 1], 0.0)


def test_generate_sf_grid_3d_mpi_periodic_all_matches_wrapped_linear_field():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]

    sf = generate_sf_grid_3d_mpi(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL", "TT", "LLL", "LTT"],
        px=1,
        boundary="periodic-all",
        comm=_FakeComm(),
    )

    assert np.isclose(sf["SF_LL_grid"][1, 0, 0], 3.0)
    assert np.isclose(sf["SF_LL_grid"][0, 1, 0], 3.0)
    assert np.isclose(sf["SF_LL_grid"][0, 0, 1], 3.0)
    assert np.isclose(sf["SF_TT_grid"][1, 0, 0], 0.0)
    assert np.isclose(sf["SF_TT_grid"][0, 1, 0], 0.0)
    assert np.isclose(sf["SF_TT_grid"][0, 0, 1], 0.0)
    assert np.isclose(sf["SF_LLL_grid"][1, 0, 0], -6.0)
    assert np.isclose(sf["SF_LLL_grid"][0, 1, 0], -6.0)
    assert np.isclose(sf["SF_LLL_grid"][0, 0, 1], -6.0)


def test_generate_sf_grid_3d_mpi_periodic_all_handles_odd_half_grid_sizes():
    x = np.arange(12, dtype=float)
    y = np.arange(10, dtype=float)
    z = np.arange(8, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u = 2.0 * xx + 0.1 * yy
    v = 3.0 * yy + 0.2 * zz
    w = 5.0 * zz + 0.3 * xx

    sf = generate_sf_grid_3d_mpi(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL"],
        px=1,
        boundary="periodic-all",
        comm=_FakeComm(),
    )

    # Regression: y shifts were silently left at zero when ny//2 was odd.
    assert sf["SF_LL_grid"][0, 1, 0] > 0.0
    assert sf["SF_LL_grid"][0, 3, 2] > 0.0
