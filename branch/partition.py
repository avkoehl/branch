import numpy as np
import numpy.ma as ma
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import skfmm

from ._io import unwrap, wrap


def allocate(mask, seeds):
    # Ordered, radius-limited claiming. Seeds claim territory in ascending
    # label order (label 1 = highest priority, e.g. the mainstem); each
    # label's reach is limited by the local shape radius, so wide branches
    # claim proportionally more space at junctions. Unclaimed remainder is
    # watershed-filled so the output labels cover the mask completely.
    mask_arr, _, meta = unwrap(mask)
    seed_arr, _, _ = unwrap(seeds)
    mask_bool = mask_arr == 1
    _check(mask_bool, seed_arr)

    radius = distance_transform_edt(mask_bool)
    allocation = np.zeros(mask_bool.shape, dtype=np.uint32)
    claimed = np.zeros(mask_bool.shape, dtype=bool)

    for label in np.unique(seed_arr[seed_arr > 0]):  # ascending = priority
        tier = (seed_arr == label) & mask_bool & ~claimed
        if not tier.any():
            continue
        phi = np.ones(mask_bool.shape)
        phi[tier] = 0.0
        obstacles = ~mask_bool | claimed
        try:
            dist, ext_radius = skfmm.extension_velocities(
                ma.MaskedArray(phi, mask=obstacles),
                radius,
                narrow=float(radius.max()) + 1.0,
            )
        except ValueError:
            continue
        reached = ~np.ma.getmaskarray(dist)
        claim = mask_bool & ~claimed & reached & (dist.data <= ext_radius.data)
        allocation[claim] = label
        claimed |= claim

    unclaimed = mask_bool & ~claimed
    if unclaimed.any() and claimed.any():
        allocation = watershed(
            image=distance_transform_edt(~claimed),
            markers=allocation,
            mask=mask_bool,
        ).astype(np.uint32)

    return wrap(allocation, meta)


def voronoi(mask, seeds):
    # Nearest-seed partition of the mask: every pixel goes to the seed label
    # it can reach by the shortest within-mask route. No ordering, no radius
    # limits. Use for simple subdivision, e.g. splitting a path's territory
    # by segment: voronoi(regions == path_id, segment_seeds).
    mask_arr, _, meta = unwrap(mask)
    seed_arr, _, _ = unwrap(seeds)
    mask_bool = mask_arr == 1
    _check(mask_bool, seed_arr)

    markers = np.where(mask_bool, seed_arr, 0).astype(np.int64)
    if not (markers > 0).any():
        return wrap(np.zeros(mask_bool.shape, dtype=np.uint32), meta)

    out = watershed(
        image=distance_transform_edt(markers == 0),
        markers=markers,
        mask=mask_bool,
    ).astype(np.uint32)
    return wrap(out, meta)


def _check(mask_bool, seed_arr):
    if seed_arr.shape != mask_bool.shape:
        raise ValueError(
            f"seeds shape {seed_arr.shape} does not match mask shape {mask_bool.shape}"
        )
    if (seed_arr[mask_bool] > 0).sum() == 0:
        raise ValueError("no seed pixels found inside the mask")
