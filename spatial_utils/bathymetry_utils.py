from typing import Tuple, Union, List

import numpy as np
from rasterio import DatasetReader

import rasterio
from shapely.geometry import LineString, Polygon, MultiLineString


def split_trajectory(base_trajectory: LineString, dataset: DatasetReader) -> Tuple[
    LineString, Union[LineString, MultiLineString]]:
    """
    Split a LineString trajectory into 2 components:
        - One intersecting the coverage area of the raster data
        - Another outside of the bounds of the raster data (can be a MultiLineString)

    The Coverage of the raster data is a box:
     p2 --------p3
     |           |
     |           |
     |           |
     p1 ------- p4
    :param base_trajectory: base surface line in x,y RD coordinates as a LineString
    :param dataset: raster Dataset

    :returns surface_line: a tuple: raster_trajectory, remaining trajectory. Where raster trajectory is the intersection
     of the base trajectory with the raster data.

    """
    raster_bounds = dataset.bounds

    p1 = [raster_bounds.left, raster_bounds.bottom]
    p2 = [raster_bounds.left, raster_bounds.top]
    p3 = [raster_bounds.right, raster_bounds.top]
    p4 = [raster_bounds.right, raster_bounds.bottom]

    raster_coverage = Polygon([p1, p2, p3, p4])  # TODO change this rough box to a convex hull or a alphashape

    raster_trajectory = base_trajectory.intersection(raster_coverage)
    remaining_trajectory = base_trajectory.symmetric_difference(raster_trajectory)
    return raster_trajectory, remaining_trajectory


def get_surface_line_from_raster(slice_line: Union[List, np.ndarray], dataset: DatasetReader) -> List[
    Tuple[float, float, float]]:
    """
    Return the surface line as a list of x,y,z in RD coordinates
    Args:
        slice_line: x,y RD coordinates of the surface line
        dataset: raster Dataset

    Returns: list

    """
    surface_line = []
    for point in slice_line:
        row_idx, col_idx = rasterio.transform.rowcol(transform=dataset.transform, xs=point[0], ys=point[1])
        arr = dataset.read(1)
        elevation = arr[row_idx - 1, col_idx - 1]
        if isinstance(elevation, float) and elevation > -1000:  # filter the empty values assigned to -9999
            surface_line.append((point[0], point[1], elevation))
    return surface_line
