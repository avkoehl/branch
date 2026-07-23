import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import branch
from branch.data import load

OUT = REPO / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def values(arr):
    return arr.values if hasattr(arr, "values") else np.asarray(arr)


def label_cmap(arr):
    labels = np.unique(arr[arr > 0])
    n = len(labels)
    idx = np.arange(n)
    hsv = np.stack(
        [
            (idx * 0.61803398875) % 1.0,
            0.55 + 0.4 * ((idx * 0.37) % 1.0),
            0.7 + 0.25 * ((idx * 0.23) % 1.0),
        ],
        axis=1,
    )
    cmap = mcolors.ListedColormap(mcolors.hsv_to_rgb(hsv))
    lookup = np.zeros(int(labels.max()) + 1, dtype=int)
    lookup[labels] = idx
    return np.ma.masked_where(arr == 0, lookup[arr]), cmap, n


def draw_labels(ax, labeled, net=None, endpoints=False, centerline=False):
    ax.imshow(mask_arr, cmap="gray_r", alpha=0.25)
    mapped, cmap, n = label_cmap(labeled)
    ax.imshow(mapped, cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
    if centerline and net is not None:
        rr, cc = np.nonzero(values(net.rasterize()))
        ax.plot(cc, rr, "k.", markersize=0.4)
    if endpoints and net is not None:
        ax.plot(net.root[1], net.root[0], "k*", markersize=14, mec="w")
        if net.tips:
            tr, tc = zip(*net.tips)
            ax.plot(tc, tr, "wo", markersize=5, mec="k")
    ax.set_axis_off()


def draw_float(ax, arr, norm, cmap="viridis", colorbar=True):
    im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="width")
    ax.set_axis_off()
    return im


def save(name, fig):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def save_labels(name, labeled, titles=None, **kw):
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_labels(ax, values(labeled), **kw)
    save(name, fig)


# -- pipeline ------------------------------------------------------------------

mask, root, tips = load()
mask_arr = values(mask) == 1
print(f"mask: {mask_arr.shape}, root: {root}, tips: {len(tips)}")

net = branch.extract(mask, root=root, tips=tips)
net_auto = branch.extract(mask, root=root)
regions = branch.allocate(mask, net.rasterize(by="path"))
regions_vor = branch.voronoi(mask, net.rasterize(by="path"))
seg_regions = branch.subdivide(regions, net)
w_lap = branch.widths(mask, net.rasterize(), method="laplace")
w_near = branch.widths(mask, net.rasterize(), method="nearest")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rw_lap = branch.region_widths(mask, net.rasterize(), regions, method="laplace")
    rw_near = branch.region_widths(mask, net.rasterize(), regions, method="nearest")


# shared discrete log-spaced bins (1-2-5 sequence) across every width image
def log_bins(vmin, vmax):
    edges, k = [], int(np.floor(np.log10(max(vmin, 1e-12))))
    while not edges or edges[-1] < vmax:
        for m in (1, 2, 5):
            edges.append(m * 10.0**k)
        k += 1
    lo = max(i for i, e in enumerate(edges) if e <= vmin) if edges[0] <= vmin else 0
    edges = [e for e in edges[lo:] if e / 1.0001 <= vmax]
    while len(edges) > 9:  # coarsen: drop the 2s, then fall back to decades
        edges = [e for e in edges if not str(e).lstrip("0.").startswith("2")]
        if len(edges) > 9:
            edges = [e for e in edges if str(e).lstrip("0.").startswith("1")]
    return edges


allw = np.concatenate([values(a).ravel() for a in (w_lap, w_near, rw_lap, rw_near)])
finite = allw[np.isfinite(allw) & (allw > 0)]
BINS = log_bins(float(finite.min()), float(finite.max()))
CMAP = plt.get_cmap("cividis", len(BINS))
NORM = mcolors.BoundaryNorm(BINS, ncolors=len(BINS), extend="max")

# -- graphical abstract ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(
    np.ma.masked_where(~mask_arr, mask_arr.astype(int)), cmap="gray", vmin=0, vmax=2
)
axes[0].plot(net.root[1], net.root[0], "k*", markersize=14, mec="w", label="root")
tr, tc = zip(*net.tips)
axes[0].plot(tc, tr, "wo", markersize=5, mec="k", label="tips")
axes[0].legend(loc="lower right", fontsize=8)
axes[0].set_title("binary branching shape + root, tips")
axes[0].set_axis_off()
mapped, cmap, n = label_cmap(values(regions))
axes[1].imshow(mask_arr, cmap="gray_r", alpha=0.25)
axes[1].imshow(mapped, cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
axes[1].set_title("branch segmentation")
axes[1].set_axis_off()
draw_float(axes[2], values(rw_lap), NORM, cmap=CMAP)
axes[2].set_title("interpolated widths")
save("abstract.png", fig)

# -- individual function outputs --------------------------------------------------

save_labels("extract_tips.png", net.rasterize(by="path"), net=net, endpoints=True)
save_labels(
    "extract_auto.png", net_auto.rasterize(by="path"), net=net_auto, endpoints=True
)

save_labels("allocate.png", regions, net=net, centerline=True)
save_labels("voronoi.png", regions_vor, net=net, centerline=True)
save_labels("subdivide.png", seg_regions, net=net, centerline=True)

# 2x2 matrix: interpolator (columns) x domain (rows), one shared colorbar
fig, axes = plt.subplots(2, 2, figsize=(14, 13))
panels = [
    (axes[0, 0], w_lap, 'widths(..., method="laplace")'),
    (axes[0, 1], w_near, 'widths(..., method="nearest")'),
    (axes[1, 0], rw_lap, 'region_widths(..., method="laplace")'),
    (axes[1, 1], rw_near, 'region_widths(..., method="nearest")'),
]
for ax, arr, title in panels:
    im = draw_float(ax, values(arr), NORM, cmap=CMAP, colorbar=False)
    ax.set_title(title, fontfamily="monospace", fontsize=11)
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="width", extend="max")
fig.savefig(OUT / "widths_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved widths_matrix.png")

# -- open_boundary: a boundary the shape is truncated by, not a real wall --------
# Mark the void just past the outlet (root side) as open. Half-widths there are
# then measured to the true flanking walls instead of collapsing at the cut edge,
# so the mainstem keeps its width down to the outlet (visible in the widths row).
rr = np.arange(mask_arr.shape[0])[:, None]
cc = np.arange(mask_arr.shape[1])[None, :]
open_boundary = (~mask_arr) & (rr >= root[0] - 10) & (cc <= root[1] + 35)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res_open = branch.analyze(mask, root, tips=tips, open_boundary=open_boundary)

fig, axes = plt.subplots(2, 2, figsize=(13, 12))
panels = [
    (axes[0, 0], "labels", regions, "regions"),
    (axes[0, 1], "labels", res_open.regions, "regions, open_boundary=…"),
    (axes[1, 0], "float", rw_lap, "widths"),
    (axes[1, 1], "float", res_open.widths, "widths, open_boundary=…"),
]
for ax, kind, arr, title in panels:
    if kind == "labels":
        mapped, cmap, n = label_cmap(values(arr))
        ax.imshow(mask_arr, cmap="gray_r", alpha=0.25)
        ax.imshow(mapped, cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
    else:
        draw_float(ax, values(arr), NORM, cmap=CMAP, colorbar=False)
    ax.contour(open_boundary.astype(int), levels=[0.5], colors="red", linewidths=0.9)
    ax.plot(root[1], root[0], "k*", markersize=12, mec="w")
    ax.set_title(title, fontfamily="monospace", fontsize=11)
    ax.set_axis_off()
fig.colorbar(
    plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
    ax=axes[1, :], fraction=0.03, pad=0.02, label="width", extend="max",
)
save("open_boundary.png", fig)
