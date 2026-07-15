import sys
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


def panel_mask(ax, mask_arr, title="", net=None):
    ax.imshow(np.ma.masked_where(mask_arr == 0, mask_arr), cmap="gray", vmin=0, vmax=2)
    if net is not None:
        ax.plot(net.root[1], net.root[0], "k*", markersize=14, mec="w", label="root")
        if net.tips:
            tr, tc = zip(*net.tips)
            ax.plot(tc, tr, "wo", markersize=5, mec="k", label="tips")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title)
    ax.set_axis_off()


def panel_labels(ax, labeled, title="", mask_arr=None, net=None, endpoints=False):
    if mask_arr is not None:
        ax.imshow(mask_arr, cmap="gray_r", alpha=0.25)
    mapped, cmap, n = label_cmap(labeled)
    ax.imshow(mapped, cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
    if net is not None and endpoints:
        ax.plot(net.root[1], net.root[0], "k*", markersize=14, mec="w")
        if net.tips:
            tr, tc = zip(*net.tips)
            ax.plot(tc, tr, "wo", markersize=5, mec="k")
    ax.set_title(title)
    ax.set_axis_off()


def panel_float(ax, arr, title="", cmap="viridis", norm=None):
    # log scale: branching shapes span orders of magnitude in width
    if norm is None:
        finite = arr[np.isfinite(arr) & (arr > 0)]
        norm = (
            mcolors.LogNorm(vmin=finite.min(), vmax=finite.max())
            if finite.size
            else None
        )
    im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="width (log scale)")
    ax.set_title(title)
    ax.set_axis_off()


# -- pipeline -----------------------------------------------------------------

mask, root, tips = load()
mask_arr = values(mask) == 1
print(f"mask: {mask_arr.shape}, root: {root}, tips: {len(tips)}")

net_tips = branch.extract(mask, root=root, tips=tips)
net_auto = branch.extract(mask, root=root)
regions = branch.allocate(mask, net_tips.rasterize(by="path"))
regions_vor = branch.voronoi(mask, net_tips.rasterize(by="path"))
w = branch.region_widths(mask, net_tips.rasterize(), regions)

# -- 1. graphical abstract: mask -> regions -> widths --------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
panel_mask(
    axes[0], mask_arr.astype(int), "binary branching shape + root, tips", net=net_tips
)
panel_labels(axes[1], values(regions), "branch segmentation")
panel_float(axes[2], values(w), "interpolated widths")
plt.tight_layout()
plt.savefig(OUT / "abstract.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved abstract.png")

# -- 2a. centerlines: with vs without tips -------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (name, net) in zip(
    axes, [("tips provided", net_tips), ("tips auto-detected", net_auto)]
):
    panel_labels(
        ax,
        values(net.rasterize(by="path")),
        f"centerline paths ({name})\n{net.segments.path_id.max()} paths, "
        f"{len(net.segments)} segments",
        mask_arr=mask_arr,
        net=net,
        endpoints=True,
    )
plt.tight_layout()
plt.savefig(OUT / "centerlines.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved centerlines.png")

# -- 2b. allocate vs voronoi (with tips) ---------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
panel_labels(
    axes[0],
    values(regions),
    "allocate (ordered, radius-limited)",
    mask_arr=mask_arr,
    net=net_tips,
)
panel_labels(
    axes[1],
    values(regions_vor),
    "voronoi (nearest centerline)",
    mask_arr=mask_arr,
    net=net_tips,
)
for ax in axes:
    cl = values(net_tips.rasterize())
    rr, cc = np.nonzero(cl)
    ax.plot(cc, rr, "k.", markersize=0.4)
plt.tight_layout()
plt.savefig(OUT / "allocation.png", dpi=150, bbox_inches="tight")
plt.close()
diff = int((values(regions) != values(regions_vor)).sum())
print(
    f"saved allocation.png ({diff} px differ, "
    f"{100 * diff / mask_arr.sum():.1f}% of mask)"
)

# -- 2c. region widths: laplace vs nearest -------------------------------------

w_nearest = branch.region_widths(mask, net_tips.rasterize(), regions, method="nearest")

both = np.concatenate([values(w).ravel(), values(w_nearest).ravel()])
finite = both[np.isfinite(both) & (both > 0)]
shared = mcolors.LogNorm(vmin=finite.min(), vmax=finite.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
panel_float(axes[0], values(w), "laplace (smooth diffusion)", norm=shared)
panel_float(axes[1], values(w_nearest), "nearest (piecewise constant)", norm=shared)
plt.tight_layout()
plt.savefig(OUT / "widths.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved widths.png")
