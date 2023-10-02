import pytest
import numpy as np

from core.data_input import Geometry
from spatial_utils.ahn_utils import SpatialUtils

TOL = 1e-2


class TestSpatialUtils:
    @pytest.mark.unittest
    def test_get_ahn_surface_line(self):
        spacial_utils = SpatialUtils()
        location_1 = Geometry(x=64663.8, y=393995.8, z=0)
        # create x,y line
        width, height = 5, 5
        slice_line = [(location_1.x + x, location_1.y) for x in list(range(width))]
        spacial_utils.get_ahn_surface_line(np.array(slice_line))
        assert len(spacial_utils.AHN_data) == 5

    @pytest.mark.unittest
    def test_get_ahn_surface_line_wrong_shape(self):
        spacial_utils = SpatialUtils()
        location_1 = Geometry(x=64663.8, y=393995.8, z=0)
        width, height = 5, 5
        slice_line = [
            (location_1.x + x, location_1.y + y, 5)
            for x in list(range(width))
            for y in list(range(height))
        ]
        with pytest.raises(ValueError) as exc_info:
            spacial_utils.get_ahn_surface_line(np.array(slice_line))
        assert (
                str(exc_info.value)
                == "The list provided should be of shape (:, 2) but is of shape (25, 3)"
        )

    def test_get_ahn3(self):
        spacial_utils = SpatialUtils()
        spacial_utils.get_ahn_surface_line(np.array([102428.7263, 472806.5402]).reshape(1, 2))
        assert np.abs(spacial_utils.AHN_data[0][2] - (-5.0669)) <= TOL
