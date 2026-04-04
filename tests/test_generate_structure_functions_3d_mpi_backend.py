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


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"sf_type": ["SS"], "scalar": None}, "scalar is required"),
        ({"boundary": None, "sf_type": ["LL"]}, "boundary='periodic-all'"),
        ({"nbins": 2, "sf_type": ["LL"]}, "nbins"),
        ({"sf_type": ["ASF_V"]}, "does not support: ASF_V"),
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
