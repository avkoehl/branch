# _io.py
from dataclasses import dataclass
import numpy as np


@dataclass
class GridMeta:
    """Everything needed to reconstruct an xr.DataArray from a numpy result."""

    coords: dict
    dims: tuple
    crs: object | None
    transform: object | None


def unwrap(arr, pixel_size=None):
    """
    Accept np.ndarray or xr.DataArray.
    Returns (values: np.ndarray, pixel_size: float, meta: GridMeta | None).
    """
    try:
        import xarray as xr

        is_xr = isinstance(arr, xr.DataArray)
    except ImportError:
        is_xr = False

    if not is_xr:
        return np.asarray(arr), float(pixel_size or 1.0), None

    meta = GridMeta(
        coords=arr.coords,
        dims=arr.dims,
        crs=getattr(arr.rio, "crs", None) if hasattr(arr, "rio") else None,
        transform=arr.rio.transform() if hasattr(arr, "rio") else None,
    )
    if pixel_size is None and meta.transform is not None:
        pixel_size = abs(
            meta.transform.a
        )  # from affine transform, assumes square pixels
    return arr.values, float(pixel_size or 1.0), meta


def wrap(values, meta):
    """np.ndarray + meta -> xr.DataArray; passthrough if meta is None."""
    if meta is None:
        return values
    import xarray as xr

    out = xr.DataArray(values, coords=meta.coords, dims=meta.dims)
    if meta.crs is not None:
        out.rio.write_crs(meta.crs, inplace=True)
        out.rio.write_transform(meta.transform, inplace=True)
    return out


def region_groups(label_arr, mask_bool=None):
    """Group a labeled raster's pixels by label, in one pass.

    Yields ``(label, flat_idx)`` per positive label, ascending, where
    ``flat_idx`` indexes the raveled grid; ``mask_bool`` optionally restricts
    which pixels count. This replaces the ``label_arr == label`` scan a
    per-region loop would otherwise do, which costs the whole grid once per
    region. The sort is stable, so each group's indices come out ascending --
    callers rely on that for ``searchsorted`` neighbour lookups and for
    reading off a bounding box.
    """
    positive = label_arr > 0
    flat = np.flatnonzero(positive if mask_bool is None else positive & mask_bool)
    if flat.size == 0:
        return
    labels = label_arr.ravel()[flat]
    order = np.argsort(labels, kind="stable")
    flat = flat[order]
    labels = labels[order]
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    for lo, hi in zip(starts, np.append(starts[1:], flat.size)):
        yield int(labels[lo]), flat[lo:hi]


def edt_field(mask_bool, open_boundary):
    """Foreground array for the local-half-width distance transform.

    ``distance_transform_edt`` of the returned array gives, at each shape pixel,
    the distance to the nearest *wall*. By default (``open_boundary is None``)
    every non-shape pixel is a wall -- the original behaviour. When
    ``open_boundary`` is given, its truthy pixels are treated as void (open,
    not a wall): they join the shape as foreground, so half-widths are measured
    only to the remaining real walls. Accepts np.ndarray or xr.DataArray.
    """
    if open_boundary is None:
        return mask_bool
    open_arr, _, _ = unwrap(open_boundary)
    open_bool = np.asarray(open_arr) > 0
    if open_bool.shape != mask_bool.shape:
        raise ValueError(
            f"open_boundary shape {open_bool.shape} does not match "
            f"mask shape {mask_bool.shape}"
        )
    field = mask_bool | open_bool
    if field.all():
        import warnings

        warnings.warn(
            "open_boundary leaves no wall pixels; local half-widths will be zero"
        )
    return field
