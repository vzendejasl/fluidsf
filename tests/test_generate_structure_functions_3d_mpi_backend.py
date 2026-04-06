import numpy as np
import pytest

from fluidsf.generate_structure_functions_3d import generate_structure_functions_3d


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

    def allreduce(self, value):
        return value


def test_generate_structure_functions_3d_mpi_backend_matches_serial():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL", "TT", "LLL", "LTT"],
        boundary="periodic-all",
    )
    parallel = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL", "TT", "LLL", "LTT"],
        boundary="periodic-all",
        backend="mpi",
        px=1,
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)


def test_generate_structure_functions_3d_mpi_backend_accepts_distributed_x_slab():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL", "TT", "LLL", "LTT"],
        boundary="periodic-all",
    )
    distributed = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LL", "TT", "LLL", "LTT"],
        boundary="periodic-all",
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == distributed.keys()
    for key in serial:
        np.testing.assert_allclose(distributed[key], serial[key], atol=1e-14)


def test_generate_structure_functions_3d_mpi_backend_matches_serial_for_scalar_sf():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]
    scalar = u + 2.0 * v

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["SS", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
    )
    parallel = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["SS", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)


def test_generate_structure_functions_3d_mpi_backend_matches_serial_for_advective_sf():
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]
    scalar = u + 2.0 * v

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S"],
        scalar=scalar,
        boundary="periodic-all",
    )
    parallel = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S"],
        scalar=scalar,
        boundary="periodic-all",
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)


def test_generate_structure_functions_3d_mpi_backend_matches_serial_with_nbins():
    x = np.arange(8, dtype=float)
    y = np.arange(8, dtype=float)
    z = np.arange(8, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]
    scalar = u + 2.0 * v

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "SS", "LLL", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
        nbins=2,
    )
    parallel = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "SS", "LLL", "LSS"],
        scalar=scalar,
        boundary="periodic-all",
        nbins=2,
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)


def test_generate_structure_functions_3d_mpi_backend_matches_serial_nonperiodic():
    x = np.arange(8, dtype=float)
    y = np.arange(8, dtype=float)
    z = np.arange(8, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]
    scalar = u + 2.0 * v

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary=None,
        nbins=2,
    )
    parallel = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary=None,
        nbins=2,
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)


def test_generate_structure_functions_3d_mpi_backend_matches_serial_mixed_periodic():
    x = np.arange(8, dtype=float)
    y = np.arange(8, dtype=float)
    z = np.arange(8, dtype=float)
    u = np.meshgrid(x, y, z, indexing="ij")[0]
    v = np.meshgrid(x, y, z, indexing="ij")[1]
    w = np.meshgrid(x, y, z, indexing="ij")[2]
    scalar = u + 2.0 * v

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary=["periodic-x", "periodic-y"],
        nbins=2,
    )
    parallel = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary=["periodic-x", "periodic-y"],
        nbins=2,
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)


def test_generate_structure_functions_3d_mpi_backend_matches_serial_asymmetric_public_layout():
    x = np.arange(128, dtype=float)
    y = np.arange(128, dtype=float)
    z = np.arange(60, dtype=float)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u = 2.0 * xx + 0.1 * yy
    v = 3.0 * yy + 0.2 * zz
    w = 5.0 * zz + 0.3 * xx

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "LL", "LLL", "LTT"],
        boundary=["periodic-x", "periodic-y"],
    )
    parallel = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["ASF_V", "LL", "LLL", "LTT"],
        boundary=["periodic-x", "periodic-y"],
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)


def test_generate_structure_functions_3d_mpi_backend_exact_sf_type_selection():
    x = np.arange(6, dtype=float)
    y = np.arange(6, dtype=float)
    z = np.arange(6, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")

    serial = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LLL"],
        boundary="periodic-all",
    )
    parallel = generate_structure_functions_3d(
        u,
        v,
        w,
        x,
        y,
        z,
        sf_type=["LLL"],
        boundary="periodic-all",
        backend="mpi",
        comm=_FakeComm(),
    )

    assert "SF_LLL_x" in serial
    assert "SF_LLL_x" in parallel
    assert "SF_LL_x" not in serial
    assert "SF_LL_x" not in parallel


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"sf_type": ["SS"], "scalar": None}, "scalar is required"),
        ({"sf_type": ["ASF_S"], "scalar": None}, "scalar is required"),
        ({"boundary": "not-a-boundary", "sf_type": ["LL"]}, "supports only boundary=None"),
    ],
)
def test_generate_structure_functions_3d_mpi_backend_rejects_unsupported_modes(
    kwargs, expected_message
):
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    z = np.arange(4, dtype=float)
    u, v, w = np.meshgrid(x, y, z, indexing="ij")

    with pytest.raises(ValueError, match=expected_message):
        generate_structure_functions_3d(
            u,
            v,
            w,
            x,
            y,
            z,
            backend="mpi",
            px=1,
            comm=_FakeComm(),
            **kwargs,
        )
