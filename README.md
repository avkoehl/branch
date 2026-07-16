# branch

Characterize binary branching shapes — rivers, valley floors, floodplains, glaciers, roots.
Given a shape mask, a root point, and (optionally) branch tips, `branch` extracts a
topology-aware centerline network, decomposes it into hierarchically ordered paths,
allocates every pixel of the shape to its path, and estimates local width everywhere.

![graphical abstract](docs/images/abstract.png)

## Install

```bash
pip install git+https://github.com/avkoehl/branch.git
```

Development (clone, then sync with dev extras):

```bash
git clone https://github.com/avkoehl/branch.git
cd branch
uv sync --extra dev
```

## Usage

```python
import branch
from branch.data import load

mask, root, tips = load()                       # bundled toy dataset

result = branch.analyze(mask, root, tips=tips)
result.network.segments                         # DataFrame: segment_id, path_id, strahler,
                                                #   length, weight, downstream_segment_id
result.regions                                  # labeled raster: each pixel -> its path
result.widths                                   # float raster: local width everywhere
```

Inputs are `np.ndarray` (with `pixel_size=`) or georeferenced `xr.DataArray`;
outputs match the input type. `root` and `tips` are `(row, col)` pixel coordinates.

## Components

`analyze` composes the functions below. Each is usable on its own.

### Centerlines

```python
net = branch.extract(mask, root, tips=tips)
```

Skeletonizes the mask, routes from each tip to the root (pruning everything else),
and decomposes the network into ordered paths — `path_id == 1` is the mainstem.

![extract with tips](docs/images/extract_tips.png)

```python
net = branch.extract(mask, root)
```

Without tips, every skeleton endpoint becomes a tip.

![extract auto tips](docs/images/extract_auto.png)

### Partitioning

```python
regions = branch.allocate(mask, net.rasterize(by="path"))
```

Assigns every pixel to a path: paths claim territory in priority order, each limited
by the local shape radius, so wide branches claim proportionally more space at junctions.

![allocate](docs/images/allocate.png)

```python
regions = branch.voronoi(mask, net.rasterize(by="path"))
```

Nearest-centerline partition — no ordering, no radius limits.

![voronoi](docs/images/voronoi.png)

```python
seg_regions = branch.subdivide(regions, net)
```

Splits each path's territory into per-segment territories.

![subdivide](docs/images/subdivide.png)

### Widths

```python
w = branch.widths(mask, net.rasterize(), method="laplace")
```

Takes exact widths (2 × distance-to-boundary) at the centerline and diffuses them
smoothly across the shape.

![widths laplace](docs/images/widths_laplace.png)

```python
w = branch.widths(mask, net.rasterize(), method="nearest")
```

Each pixel takes the width of its nearest centerline pixel (piecewise constant).

![widths nearest](docs/images/widths_nearest.png)

```python
w = branch.region_widths(mask, net.rasterize(), regions)
```

Interpolates independently within each region, so widths never diffuse across path
boundaries at junctions. Supports both methods.

![region widths](docs/images/region_widths.png)
