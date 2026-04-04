"""Public wrapper for the Phase 1 MPI 3D SF-grid implementation."""

from .mpi.generate_sf_3d_mpi import generate_sf_grid_3d_mpi

__all__ = ("generate_sf_grid_3d_mpi",)
