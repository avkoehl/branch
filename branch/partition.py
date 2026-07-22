import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.segmentation import watershed

from ._io import unwrap, wrap

SQRT2 = np.sqrt(2.0)
# forward-only neighbour offsets; directed=False makes each bidirectional
_EDGES = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)]


def allocate(mask, seeds):
    # Ordered, radius-limited claiming. Each path (seed label) claims the mask
    # pixels within *some* of its seeds' local half-width, measured as a
    # boundary-respecting (geodesic) distance -- so a wide-but-farther seed can
    # reach a pixel a narrow-but-nearer seed cannot. Paths are processed
    # biggest-first (label 1 = mainstem = highest priority) and a pixel is kept
    # by the first (biggest) path to reach it, so wide branches claim
    # proportionally more space at junctions. Unclaimed remainder is
    # watershed-filled so the labels cover the mask completely.
    #
    # Each path's claim is a windowed, radius-bounded Dijkstra (see _reach), so
    # cost is O(tube area) per path rather than O(domain); the mask never
    # changes during the loop, so there is no per-step graph rebuild.
    mask_arr, _, meta = unwrap(mask)
    seed_arr, _, _ = unwrap(seeds)
    mask_bool = mask_arr == 1
    _check(mask_bool, seed_arr)

    H, W = mask_bool.shape
    radius = distance_transform_edt(mask_bool)  # local half-width per pixel
    allocation = np.zeros(mask_bool.shape, dtype=np.uint32)

    # group seed pixels by label once, so each path works from its own coords
    # (and bbox) instead of scanning the full seed array every iteration
    flat = np.flatnonzero(seed_arr)
    labels_flat = seed_arr.ravel()[flat]
    order = np.argsort(labels_flat, kind="stable")
    flat = flat[order]
    labels_sorted = labels_flat[order]
    uniq, starts = np.unique(labels_sorted, return_index=True)  # ascending
    bounds = np.append(starts, labels_sorted.size)

    for i, label in enumerate(uniq):  # ascending label == biggest path first
        idx = flat[bounds[i]:bounds[i + 1]]
        rr, cc = idx // W, idx % W
        keep = mask_bool[rr, cc]
        if not keep.any():
            continue
        rr, cc = rr[keep], cc[keep]
        rad = radius[rr, cc]
        R = float(rad.max())  # farthest this path can reach = search bound

        pad = int(np.ceil(R)) + 1
        r0, r1 = max(int(rr.min()) - pad, 0), min(int(rr.max()) + pad + 1, H)
        c0, c1 = max(int(cc.min()) - pad, 0), min(int(cc.max()) + pad + 1, W)

        seed_local = np.stack([rr - r0, cc - c0], axis=1)
        tube = _reach(mask_bool[r0:r1, c0:c1], seed_local, rad, R)
        sub = allocation[r0:r1, c0:c1]
        sub[tube & (sub == 0)] = label  # keep only where no bigger path won

    claimed = allocation > 0
    unclaimed = mask_bool & ~claimed
    if unclaimed.any() and claimed.any():
        allocation = watershed(
            image=distance_transform_edt(~claimed),
            markers=allocation,
            mask=mask_bool,
        ).astype(np.uint32)

    return wrap(allocation, meta)


def _reach(mask_win, seed_rc, radii, R):
    # Pixels within some seed's local half-width, geodesically, inside the
    # window. A virtual super-source is wired to each seed s with edge weight
    # R - radius(s) (>= 0), so one bounded search from it gives, per pixel q,
    #   dist_V(q) = R + min_s ( geodist(q, s) - radius(s) ),
    # and dist_V(q) <= R is exactly "some seed's radius reaches q". The bound
    # (limit=R) keeps the search inside the tube. Metric is octile ({1, sqrt2}),
    # matching centerline._dijkstra_tree.
    h, w = mask_win.shape
    ys, xs = np.nonzero(mask_win)
    M = ys.size
    if M == 0:
        return np.zeros((h, w), dtype=bool)
    ids = np.full((h, w), -1, dtype=np.int64)  # pixel -> node id, -1 off-mask
    ids[ys, xs] = np.arange(M)
    V = M  # super-source node id

    rows, cols, wts = [], [], []
    for dr, dc, step in _EDGES:  # grid edges between adjacent in-mask pixels
        ny, nx = ys + dr, xs + dc
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nb = np.where(ok, ids[np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)], -1)
        keep = nb >= 0
        rows.append(ids[ys[keep], xs[keep]])
        cols.append(nb[keep])
        wts.append(np.full(int(keep.sum()), step))

    sids = ids[seed_rc[:, 0], seed_rc[:, 1]]  # super-source -> each seed
    ok = sids >= 0
    rows.append(np.full(int(ok.sum()), V))
    cols.append(sids[ok])
    wts.append(R - radii[ok])

    n = M + 1
    graph = csr_matrix(
        (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    dist = dijkstra(graph, directed=False, indices=V, limit=R)
    out = np.zeros((h, w), dtype=bool)
    out[ys, xs] = dist[:M] <= R
    return out


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
