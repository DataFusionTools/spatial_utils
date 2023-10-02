from dataclasses import dataclass, asdict, field
import concurrent.futures
import requests
import numpy as np
from typing import List, Union
from enum import Enum


class DataTypeAhn(Enum):
    DTM = "dtm"
    DSM = "dsm"


@dataclass
class SpatialUtils:
    """Spatial utils class."""

    ahn_type: str = "ahn3"
    data_type: str = DataTypeAhn.DTM.value
    url_ahn: str = field(init=False)
    request: str = "GetFeatureInfo"
    service: str = "WMS"
    crs: str = "EPSG:28992"
    response_crs: str = "EPSG:28992"
    width: str = "4000"
    height: str = "4000"
    info_format: str = "application/json"
    version: str = "1.3.0"
    layers: str = field(init=False)
    query_layers: str = field(init=False)
    bbox: str = ""
    i: str = "2000"
    j: str = "2000"
    max_workers: int = 500
    AHN_data: np.ndarray = np.empty((0, 3))
    surface_line: List = field(default_factory=list) #TODO to be simplified or clarified, why are each element made of 4 elements?

    def __post_init__(self):
        """
        Post initialisation.
        Check type of AHN and define the fields accordingly.
        """
        self.url_ahn = (
            f"https://service.pdok.nl/rws/{self.ahn_type}/wms/v1_0"
        )
        if self.ahn_type == "ahn2":
            self.layers = f"{self.ahn_type}_5m"
            self.query_layers = f"{self.ahn_type}_5m"
        else:
            self.layers = f"{self.ahn_type}_05m_{self.data_type}"
            self.query_layers = f"{self.ahn_type}_05m_{self.data_type}"

    @property
    def dictionary_parameters(self):
        dictionary_parameters = asdict(self)
        dictionary_parameters.pop("url_ahn")
        dictionary_parameters.pop("ahn_type")
        dictionary_parameters.pop("AHN_data")
        dictionary_parameters.pop("surface_line")

        return dictionary_parameters

    @staticmethod
    def get_ahn_value_from_response(response):
        if response.status_code == 200:
            response_dict = response.json()
            features = response_dict.get("features")
            if bool(features):
                properties = features[0].get("properties")
                if properties:
                    return float(properties.get('value_list'))
            elif features == []:
                return None
            raise ValueError("The returned dictionary was not in the correct format.")
        else:
            raise ConnectionError(
                "Connection with the https://service.pdok.nl was not successful. "
            )

    def get_ahn_surface_line(self, slice_line: Union[List, np.ndarray]):
        """Function that returns a list with the x,y coordinate and elevation.

        :param slice_line: A list containing points.
        :returns surface_line: list with the x, y coordinate and the elevation.
        """

        # Check dimensions of the inputted list
        is_slice_line_2d_array = len(np.array(slice_line).shape) == 2
        does_slice_line_have_2d_points = np.array(slice_line).shape[-1] == 2
        if not (is_slice_line_2d_array and does_slice_line_have_2d_points):
            raise ValueError(
                f"The list provided should be of shape (:, 2) but is of shape {np.array(slice_line).shape}"
            )

        # concurrency: read the AHN
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
        ) as executor:
            results = [
                executor.submit(self.collect_AHN_data, point, i)
                for i, point in enumerate(slice_line)
            ]

        executor.shutdown()

        # catch exceptions from concurrency
        for result in results:
            if result.exception() is not None:
                raise result.exception()

        # sort AHN points order
        self.AHN_data = np.array(self.surface_line)[
                            np.array(self.surface_line)[:, 0].argsort()
                        ][:, 1:]
        max_allowed_elevation = 323
        min_allowed_elevation = -7
        self.AHN_data = np.array([point for point in self.AHN_data if point[-1] <= max_allowed_elevation and point[-1] >= min_allowed_elevation])
        return self.surface_line #TODO: consider using getter and setter for surface_line? The initial function was actually NOT a getter because nothing was returned

    def collect_AHN_data(self, point, idx):
        """

        :param point:
        :param idx:
        """
        self.bbox = f"{point[0] - 1000}, {point[1] - 1000}, {point[0] + 1000}, {point[1] + 1000}"
        response = requests.get(self.url_ahn, params=self.dictionary_parameters)
        ahn_value = self.get_ahn_value_from_response(response)
        if ahn_value is not None:
            self.surface_line.append([idx, point[0], point[1], ahn_value])
