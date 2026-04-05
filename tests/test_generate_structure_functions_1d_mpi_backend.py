import numpy as np

from fluidsf.generate_structure_functions_1d import generate_structure_functions_1d


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


def test_generate_structure_functions_1d_mpi_backend_matches_serial_nonperiodic():
    x = np.arange(8, dtype=float)
    u = x.copy()
    v = 2.0 * x
    scalar = 3.0 * x

    serial = generate_structure_functions_1d(
        u,
        x,
        sf_type=["LL", "TT", "SS", "LLL", "LTT", "LSS"],
        v=v,
        scalar=scalar,
        boundary=None,
        nbins=2,
    )
    parallel = generate_structure_functions_1d(
        u,
        x,
        sf_type=["LL", "TT", "SS", "LLL", "LTT", "LSS"],
        v=v,
        scalar=scalar,
        boundary=None,
        nbins=2,
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)


def test_generate_structure_functions_1d_mpi_backend_matches_serial_periodic_latlon():
    x = np.arange(8, dtype=float)
    y = np.arange(8, dtype=float)
    u = x.copy()
    v = 2.0 * x
    scalar = 3.0 * x

    serial = generate_structure_functions_1d(
        u,
        x,
        sf_type=["LL", "TT", "SS", "LLL", "LTT", "LSS"],
        v=v,
        y=y,
        scalar=scalar,
        boundary="Periodic",
        grid_type="latlon",
        nbins=2,
    )
    parallel = generate_structure_functions_1d(
        u,
        x,
        sf_type=["LL", "TT", "SS", "LLL", "LTT", "LSS"],
        v=v,
        y=y,
        scalar=scalar,
        boundary="Periodic",
        grid_type="latlon",
        nbins=2,
        backend="mpi",
        comm=_FakeComm(),
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        np.testing.assert_allclose(parallel[key], serial[key], atol=1e-14)
