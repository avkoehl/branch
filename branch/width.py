import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from ._io import unwrap, wrap

SQRT2 = np.sqrt(2.0)


def widths(mask, centerline, method="laplace", pixel_size=None):
    # Per-pixel width of the shape. Exact widths (2 * distance-to-boundary)
    # are taken at centerline pixels and interpolated across the mask.
    # method="laplace": smooth diffusion (Laplace equation, Dirichlet BCs at
    #   the centerline) — continuous fields, best for downstream analysis.
    # method="nearest": each pixel takes the width of its nearest centerline
    #   pixel (a Voronoi-style assignment, cf. branch.voronoi) — piecewise
    #   constant, fast, exact at the centerline.
    if method not in ("laplace", "nearest"):
        raise ValueError(f"method must be 'laplace' or 'nearest', got {method!r}")

    mask_arr, px, meta = unwrap(mask, pixel_size)
    cl_arr, _, _ = unwrap(centerline)
    mask_bool = mask_arr == 1
    cl_bool = (cl_arr > 0) & mask_bool
    if cl_arr.shape != mask_bool.shape:
        raise ValueError(
            f"centerline shape {cl_arr.shape} does not match mask shape {mask_bool.shape}"
        )
    if not cl_bool.any():
        raise ValueError("no centerline pixels found inside the mask")

    seed_widths = np.where(cl_bool, distance_transform_edt(mask_bool) * px * 2.0, 0.0)

    interp = _laplace if method == "laplace" else _nearest
    result = interp(cl_bool, mask_bool, seed_widths)

    out = np.where(mask_bool, result, np.nan)
    out = wrap(out, meta)
    if meta is not None:
        try:
            out.rio.write_nodata(np.nan, inplace=True)
        except AttributeError:
            pass
    return out


def region_widths(mask, centerline, regions, method="laplace", pixel_size=None):
    # Like widths(), but interpolated independently within each labeled
    # region (e.g. the output of branch.allocate), so widths do not diffuse
    # across path boundaries at junctions. Each region is seeded only by the
    # centerline pixels inside it. Regions containing no centerline pixels
    # are filled by nearest-centerline fallback (with a warning) so the
    # output always covers the mask.
    if method not in ("laplace", "nearest"):
        raise ValueError(f"method must be 'laplace' or 'nearest', got {method!r}")

    mask_arr, px, meta = unwrap(mask, pixel_size)
    cl_arr, _, _ = unwrap(centerline)
    reg_arr, _, _ = unwrap(regions)
    mask_bool = mask_arr == 1
    cl_bool = (cl_arr > 0) & mask_bool
    if cl_arr.shape != mask_bool.shape or reg_arr.shape != mask_bool.shape:
        raise ValueError("mask, centerline, and regions must share one shape")
    if not cl_bool.any():
        raise ValueError("no centerline pixels found inside the mask")

    seed_widths = np.where(cl_bool, distance_transform_edt(mask_bool) * px * 2.0, 0.0)
    interp = _laplace if method == "laplace" else _nearest

    out = np.full(mask_bool.shape, np.nan)
    for label in np.unique(reg_arr[(reg_arr > 0) & mask_bool]):
        region = (reg_arr == label) & mask_bool
        region_cl = cl_bool & region
        if not region_cl.any():
            continue  # filled by fallback below
        result = interp(region_cl, region, seed_widths)
        out[region] = result[region]

    leftover = mask_bool & np.isnan(out)
    if leftover.any():
        warnings.warn(
            f"{int(leftover.sum())} mask pixels fall in regions with no "
            "centerline pixels (or outside any region); filled by nearest "
            "centerline width"
        )
        fallback = _nearest(cl_bool, mask_bool, seed_widths)
        out[leftover] = fallback[leftover]

    out = np.where(mask_bool, out, np.nan)
    out = wrap(out, meta)
    if meta is not None:
        try:
            out.rio.write_nodata(np.nan, inplace=True)
        except AttributeError:
            pass
    return out


def _nearest(cl_bool, mask_bool, seed_widths):
    _, idx = distance_transform_edt(~cl_bool, return_indices=True)
    return seed_widths[idx[0], idx[1]]


def _laplace(cl_bool, mask_bool, seed_widths):
    n = int(mask_bool.sum())
    idx_map = np.full(mask_bool.shape, -1, dtype=np.int64)
    idx_map[mask_bool] = np.arange(n)

    yy, xx = np.nonzero(mask_bool)
    ids = idx_map[yy, xx]
    is_seed = cl_bool[yy, xx]

    rows, cols, data = [], [], []
    b = np.zeros(n)

    # Dirichlet BCs at centerline pixels
    seed_ids = ids[is_seed]
    rows.append(seed_ids)
    cols.append(seed_ids)
    data.append(np.ones(len(seed_ids)))
    b[seed_ids] = seed_widths[yy[is_seed], xx[is_seed]]

    # 8-connected Laplace stencil elsewhere (diagonals weighted 1/sqrt(2) for
    # isotropy; also prevents isolating pixels only connected diagonally)
    fy, fx, fids = yy[~is_seed], xx[~is_seed], ids[~is_seed]
    weight_sum = np.zeros(len(fids))
    offsets = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1 / SQRT2),
        (-1, 1, 1 / SQRT2),
        (1, -1, 1 / SQRT2),
        (1, 1, 1 / SQRT2),
    ]
    h, w = mask_bool.shape
    for dy, dx, wt in offsets:
        ny, nx = fy + dy, fx + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        has = np.zeros(len(fids), dtype=bool)
        has[ok] = mask_bool[ny[ok], nx[ok]]
        rows.append(fids[has])
        cols.append(idx_map[ny[has], nx[has]])
        data.append(np.full(int(has.sum()), wt))
        weight_sum[has] += wt

    rows.append(fids)
    cols.append(fids)
    data.append(-weight_sum)

    A = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    x0 = seed_widths[yy, xx]
    x, info = cg(A, b, x0=x0, rtol=1e-4)
    if info != 0:
        warnings.warn("conjugate gradient solver did not converge")

    out = np.zeros(mask_bool.shape)
    out[mask_bool] = x
    return out
