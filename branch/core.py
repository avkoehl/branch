from dataclasses import dataclass

import numpy as np

from ._io import unwrap, wrap
from .centerline import extract, Network
from .partition import allocate, voronoi
from .width import region_widths


@dataclass
class Result:
    network: Network
    regions: object  # labeled raster, path- or segment-level per analyze(level=)
    widths: object  # float raster, NaN outside mask


def subdivide(regions, network: Network):
    # Subdivide each path's territory (from allocate) into segment-level
    # territories. Each territory is seeded only by its own path's segments,
    # so neighboring paths' labels (e.g. shared junction pixels) never bleed
    # across boundaries. Pixels in territories whose path has no segments in
    # the table remain 0.
    reg_arr, _, meta = unwrap(regions)
    if reg_arr.shape != network.shape:
        raise ValueError(
            f"regions shape {reg_arr.shape} does not match network grid {network.shape}"
        )

    out = np.zeros(network.shape, dtype=np.uint32)
    for path_id, group in network.segments.groupby("path_id"):
        territory = (reg_arr == path_id).astype(np.uint8)
        if not territory.any():
            continue  # path swallowed during allocation
        seeds = np.zeros(network.shape, dtype=np.uint32)
        for _, row in group.iterrows():
            rc = np.asarray(row["pixels"])
            seeds[rc[:, 0], rc[:, 1]] = row["segment_id"]
        seeds = np.where(territory == 1, seeds, 0)
        if not (seeds > 0).any():
            continue
        sub = voronoi(territory, seeds)
        sub = np.asarray(sub)
        out[sub > 0] = sub[sub > 0]

    return wrap(out, meta)


def analyze(
    mask,
    root,
    tips=None,
    path_by="area",
    level="path",
    width_method="laplace",
    pixel_size=None,
    open_boundary=None,
) -> Result:
    # Full pipeline: extract -> allocate -> (subdivide) -> region_widths.
    if level not in ("path", "segment"):
        raise ValueError(f"level must be 'path' or 'segment', got {level!r}")

    network = extract(
        mask, root, tips=tips, path_by=path_by, pixel_size=pixel_size,
        open_boundary=open_boundary,
    )
    regions = allocate(mask, network.rasterize(by="path"), open_boundary=open_boundary)
    if level == "segment":
        regions = subdivide(regions, network)
    w = region_widths(
        mask,
        network.rasterize(),
        regions,
        method=width_method,
        pixel_size=pixel_size,
        open_boundary=open_boundary,
    )
    return Result(network=network, regions=regions, widths=w)
