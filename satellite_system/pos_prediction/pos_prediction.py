from datetime import datetime
import numpy as np
import numpy.typing as npt
from ..groups.group import Group
from satellite_system.utils.constants import EARTH_ANGULAR_VELOCITY_RAD
from satellite_system.utils.coord_converter import *


SatGeomData_dtype = np.dtype([
    ("lat", float),
    ("lon", float),
    ("height", float),
    ("view_angle", float),
    ("reg_number", int),
])


class PosPrediction:
    def __init__(self, group: Group):
        self.group = group
        self.cache = {}

    def predict_positions(self, target_time: datetime
                          ) -> npt.NDArray[SatGeomData_dtype]:
        if target_time in self.cache:
            return self.cache[target_time]

        results = []
        for orbit in self.group.orbits:
            phase_coords = np.empty(len(orbit.satellites),
                                    dtype=PhaseCoord_dtype)

            phase_coords["longitude_asc"] = (
                orbit.longitude_asc
                - np.degrees(EARTH_ANGULAR_VELOCITY_RAD
                             * (target_time - orbit.creat_date).total_seconds())
                + 360
            ) % 360

            phase_coords["inclin"] = orbit.inclin
            phase_coords["phase_on_orbit"] = np.array(
                [satellite.phase for satellite in orbit.satellites]
            )
            phase_coords["height"] = orbit.height

            geo_coords = CoordConverter().phase_to_geo_np(phase_coords)

            orbit_result = np.empty(len(geo_coords), dtype=SatGeomData_dtype)
            orbit_result["lat"] = geo_coords["lat"]
            orbit_result["lon"] = geo_coords["lon"]
            orbit_result["height"] = geo_coords["height"]
            orbit_result["view_angle"] = np.array(
                [satellite.view_angle for satellite in orbit.satellites]
            )
            orbit_result["reg_number"] = np.array(
                [satellite.reg_number for satellite in orbit.satellites]
            )
            results.append(orbit_result)

        result = np.concatenate(results)
        self.cache[target_time] = result
        return result