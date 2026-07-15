from dataclasses import dataclass

import xarray as xr

from .centerlines import get_centerlines, Centerlines
from .allocation import get_allocation, subdivide_paths_into_segments
from .widths import get_widths
from .helpers import flowlines_to_endpoints, fill_holes

# change
# operate on a single network at a time. if user wants to process multiple networks, they can call the function multiple times.
# inputs:
# - mask: 2D boolean array where True indicates the presence of the valley (or river) and False indicates the absence.
# - inlets: list of tuples, where each tuple contains a list of (row, col) coordinates representing the inlet points and a tuple representing the outlet point (row, col).
# - outlet: tuple (row, col) representing the outlet point of the valley (or river).
# - width_method: string indicating the method to use for width calculation. Options could include "laplace", "xsection"
# - inlet_distance_threshold

# outputs
# centerlines: Centerlines object containing the centerline raster and per-segment annotations (segment_id, network_id, path_label, path_uid, strahler_order, downstream_segment_id, length, pixels).
# allocation: 2D array where each pixel is assigned a segment ID corresponding to the nearest centerline segment.
# widths: 2D array where each pixel has a width value corresponding to the width of the valley at that location, calculated using the specified method.

# usage
# valley_result = measure_valley(mask, inlets, outlet, width_method="laplace", inlet_distance_threshold=100.0)
# centerlines = valley_result.centerlines
# allocation = valley_result.allocation
# widths = valley_result.widths


@dataclass
class ValleyResult:
    centerlines: Centerlines
    allocation: xr.DataArray
    widths: xr.DataArray


def measure_valley(
    mask: xr.DataArray,
    networks: list[tuple[list[tuple[int, int]], tuple[int, int]]],
    width_method: str = "laplace",
    inlet_distance_threshold: float = 100.0,
) -> ValleyResult:
    """
    Full pipeline: centerlines → segment allocation → widths.

    See individual functions for details on each step.
    """
    centerlines = get_centerlines(
        mask, networks, inlet_distance_threshold=inlet_distance_threshold
    )
    allocation = get_allocation(centerlines, mask)
    widths = get_widths(centerlines, mask, allocation=allocation, method=width_method)
    return ValleyResult(centerlines=centerlines, allocation=allocation, widths=widths)


__all__ = [
    "measure_valley",
    "get_centerlines",
    "get_allocation",
    "get_widths",
    "subdivide_paths_into_segments",
    "flowlines_to_endpoints",
    "fill_holes",
    "Centerlines",
    "ValleyResult",
]
