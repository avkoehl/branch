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
