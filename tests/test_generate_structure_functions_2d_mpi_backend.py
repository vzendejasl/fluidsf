import numpy as np

from fluidsf.generate_structure_functions_2d import generate_structure_functions_2d


class _FakeComm:
    def __init__(self):
        self._rank = 0
        self._size = 1

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def allreduce(self, value):
        return value


def test_generate_structure_functions_2d_mpi_backend_matches_serial_nonperiodic():
    x = np.arange(8, dtype=float)
    y = np.arange(8, dtype=float)
    u = np.meshgrid(x, y)[0]
    v = 0.5 * np.meshgrid(x, y)[0]
    scalar = 0.25 * np.meshgrid(x, y)[0]

    serial = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary=None,
        nbins=2,
    )
    parallel = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
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


def test_generate_structure_functions_2d_mpi_backend_matches_serial_mixed_periodic():
    x = np.arange(8, dtype=float)
    y = np.arange(8, dtype=float)
    u = np.meshgrid(x, y)[0]
    v = 0.5 * np.meshgrid(x, y)[0]
    scalar = 0.25 * np.meshgrid(x, y)[0]

    serial = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-x",
        nbins=2,
    )
    parallel = generate_structure_functions_2d(
        u,
        v,
        x,
        y,
        sf_type=["ASF_V", "ASF_S", "LL", "TT", "SS", "LLL", "LTT", "LSS"],
        scalar=scalar,
        boundary="periodic-x",
        nbins=2,
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)
