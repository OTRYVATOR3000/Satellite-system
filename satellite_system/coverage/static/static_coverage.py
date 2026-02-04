from typing import List, Set, Tuple
import math
import numpy as np
import numpy.typing as npt
from datetime import *
from satellite_system.pos_prediction.pos_prediction import *
from satellite_system.groups.group import Group
from satellite_system.utils.constants import EARTH_RADIUS, EARTH_SQUARE_KM2
from satellite_system.utils.hex_manager import HexManager
from satellite_system.utils.coord_converter import  *
from numpy.lib.recfunctions import structured_to_unstructured


CenterCoord_dtype = np.dtype([
    ("lat", float),
    ("lon", float),
    ("cell", "U15")
])


class StaticCoverage:
    def __init__(self, group: Group, resolution: int):
        self.group = group
        self.resolution = resolution
        self.cache: dict[datetime, float] = {}
        self.aperture_angles: npt.NDArray[float] = None
        self.centers: npt.NDArray[CenterCoord_dtype] = None

    def preprocess(self):
        centers_list = HexManager().get_centers(self.resolution)
        centers_list.sort(key = lambda x: x[0])
        self.centers = np.array(centers_list, dtype=CenterCoord_dtype)

        heights = np.array([orbit.height for orbit in self.group.orbits for _ in orbit.satellites])
        view_angles = np.array([satellite.view_angle for orbit in self.group.orbits for satellite in orbit.satellites])
        view_angles_rad = np.radians(view_angles)

        R = EARTH_RADIUS
        h = heights
        theta = view_angles_rad

        betta = np.arcsin((R + h) / R * np.sin(theta))
        self.aperture_angles = np.degrees(betta - theta)

    def get_covered_centers_union(self, target_time: datetime) -> npt.NDArray[CenterCoord_dtype]:
        if self.centers is None:
            self.preprocess()

        result = np.empty(0, dtype=CenterCoord_dtype)

        predictor = PosPrediction(self.group)
        positions = predictor.predict_positions(target_time)

        min_lats = np.clip(positions["lat"] - self.aperture_angles, -90, 90)
        max_lats = np.clip(positions["lat"] + self.aperture_angles, -90, 90)
        lat_rad = np.radians(positions["lat"])
        lon_half_width = self.aperture_angles / np.maximum(np.cos(lat_rad), 1e-6)

        min_lons = (positions["lon"] - lon_half_width) % 360
        max_lons = (positions["lon"] + lon_half_width) % 360

        center_lats = self.centers["lat"]

        for i in range(len(positions)):
            lat_min = min_lats[i]
            lat_max = max_lats[i]
            lon_min = min_lons[i]
            lon_max = max_lons[i]
            lower_bound = np.searchsorted(center_lats, lat_min, side="left")
            upper_bound = np.searchsorted(center_lats, lat_max, side="right")
            candidate_points = self.centers[lower_bound:upper_bound]

            center_lons = candidate_points["lon"] % 360
            if lon_min <= lon_max:
                lon_mask = (center_lons >= lon_min) & (center_lons <= lon_max)
            else:
                lon_mask = (center_lons >= lon_min) | (center_lons <= lon_max)
            candidate_points = candidate_points[lon_mask]

            sat_structured_np = CoordConverter().geo_to_dec_single(positions["lat"][i], positions["lon"][i], positions["height"][i])
            sat_np = structured_to_unstructured(sat_structured_np[['x', 'y', 'z']]).T


            candidates_structured_np = CoordConverter().geo_2d_to_dec_np(candidate_points)
            candidates_coords = structured_to_unstructured(candidates_structured_np[['x', 'y', 'z']])

            cos_center = (candidates_coords @ sat_np) / np.linalg.norm(candidates_coords, axis=1, keepdims=True) / np.linalg.norm(sat_np)
            cos_aperture_angle = np.cos(np.radians(self.aperture_angles[i]))
            final_mask = (cos_aperture_angle <= cos_center).ravel()

            result = np.concatenate((result, candidate_points[final_mask]))

        return np.unique(result)

    def calculate_coverage(self, target_time: datetime) -> float:
        if target_time in self.cache:
            return self.cache[target_time]
        return HexManager().get_square(self.get_covered_centers_union(target_time)["cell"]) / (4 * math.pi * EARTH_RADIUS**2)
