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


def draw_float(ax, arr, norm):
    im = ax.imshow(arr, cmap="viridis", norm=norm, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="width (log scale)")
    ax.set_axis_off()


def save(name, fig):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def save_labels(name, labeled, titles=None, **kw):
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_labels(ax, values(labeled), **kw)
    save(name, fig)


def save_float(name, arrs, titles, norm):
    arrs = [values(a) for a in arrs]
    fig, axes = plt.subplots(1, len(arrs), figsize=(8 * len(arrs), 7), squeeze=False)
    for ax, a, t in zip(axes[0], arrs, titles):
        draw_float(ax, a, norm)
        ax.set_title(t)
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

# shared log scale across every width image so they are directly comparable
allw = np.concatenate([values(a).ravel() for a in (w_lap, w_near, rw_lap, rw_near)])
finite = allw[np.isfinite(allw) & (allw > 0)]
NORM = mcolors.LogNorm(vmin=finite.min(), vmax=finite.max())

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
draw_float(axes[2], values(rw_lap), NORM)
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

save_float("widths_laplace.png", [w_lap], [""], NORM)
save_float("widths_nearest.png", [w_near], [""], NORM)
save_float(
    "region_widths.png",
    [rw_lap, rw_near],
    ['method="laplace"', 'method="nearest"'],
    NORM,
)
