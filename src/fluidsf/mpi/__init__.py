"""MPI helpers for FluidSF."""

from .generate_sf_3d_mpi import generate_sf_grid_3d_mpi
from .slab_decomp_3d import (
    compute_slab_bounds_1d,
    generate_sf_grid_3d_periodic_z_slab_mpi,
    compute_velocity_sf_reduction_3d_periodic_z_slab_mpi,
    exchange_periodic_halo_z,
    extract_local_z_slab,
)
from .separation_map import (
    compute_axis_index_list,
    compute_rank_coordinates,
    compute_separation_map,
    compute_separation_pairs_for_rank,
    validate_processor_grid,
)

__all__ = (
    "generate_sf_grid_3d_mpi",
    "compute_slab_bounds_1d",
    "generate_sf_grid_3d_periodic_z_slab_mpi",
    "compute_axis_index_list",
    "compute_rank_coordinates",
    "compute_separation_map",
    "compute_separation_pairs_for_rank",
    "compute_velocity_sf_reduction_3d_periodic_z_slab_mpi",
    "exchange_periodic_halo_z",
    "extract_local_z_slab",
    "validate_processor_grid",
)
