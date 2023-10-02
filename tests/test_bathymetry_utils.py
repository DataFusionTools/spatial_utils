from pathlib import Path

import numpy as np
import rasterio
import shapefile
from shapely.geometry import LineString

from spatial_utils.GWS_RD_conversion import GWSRDConvertor
from spatial_utils.ahn_utils import SpatialUtils
from spatial_utils.bathymetry_utils import split_trajectory, \
    get_surface_line_from_raster


class TestRasterUtils():

    def test_raster_bathymetry_case(self):
        """Test Case for Prumer. The shapefile contains a Polyline intersecting the area covered in the raster."""
        path_raster = Path(__file__).parent / "test_input/bathymetry_data" / "clipped_grey.tif"
        path_shp = Path(__file__).parent / "test_input/bathymetry_data" / "dense_trajectory.zip"
        with rasterio.open(path_raster) as src:
            shapefile_obj = shapefile.Reader(path_shp)

            # Careful, coordinates from shapefile are provided as lon/lat!
            line_coords = shapefile_obj.shape(0).points  # lon/lat
            base_traj = LineString([GWSRDConvertor().to_rd(latin=pt[1], lonin=pt[0]) for pt in line_coords])

            # split the base linestring trajectory into 2 parts
            raster_traj, ahn_traj = split_trajectory(base_trajectory=base_traj, dataset=src)

            # Get raster data in an array
            slice_line = [[x, y] for x, y in zip(*raster_traj.coords.xy)]
            surface_line_raster = get_surface_line_from_raster(slice_line, src)
            raster_data = np.array(surface_line_raster)

            # Get AHN data in an array
            spatial_utils = SpatialUtils()
            slice = []
            for linestring in ahn_traj.geoms:
                slice.extend([[x, y] for x, y in zip(*linestring.coords.xy)])
            spatial_utils.get_ahn_surface_line(slice)

            all_data = np.concatenate((raster_data, spatial_utils.AHN_data), axis=0)

            raster_data_expected = [[126997.91783334357, 502058.60380855144, -1.8], [126998.60589887603, 502056.00828839355, -2.37],
                       [126998.9474719857, 502054.27860882354, -2.43], [126999.64045844342, 502052.54692789726, -2.49],
                       [126999.63881852772, 502052.2589815421, -2.51], [127000.15445901576, 502050.2403550331, -2.53],
                       [127000.32524602536, 502049.37551530893, -2.55], [127001.19230058382, 502047.3548876089, -2.61],
                       [127001.54043445105, 502046.7769936558, -2.62], [127002.40912973021, 502045.0443125337, -2.58],
                       [127003.27290653318, 502042.4477925223, -2.46], [127003.96753589055, 502041.0040586007, -2.42],
                       [127004.66216566082, 502039.5603247814, -2.23], [127005.35843541854, 502038.4045374133, -1.85],
                       [127005.7049308674, 502037.5386974204, -1.57], [127006.22549403447, 502036.3839106522, -1.15],
                       [127006.74769694117, 502035.51707028935, -0.88]]
            assert raster_data_expected == raster_data.tolist()
