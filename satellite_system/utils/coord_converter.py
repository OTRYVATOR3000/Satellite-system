from .singleton import singleton
import numpy as np
import numpy.typing as npt
from satellite_system.utils.constants import EARTH_RADIUS
from collections import namedtuple

GeoCoord_dtype_cell = np.dtype([
    ("lat", float),
    ("lon", float),
    ("height", float),
    ("cell", "U15")
])

GeoCoord_dtype = np.dtype([
    ("lat", float),
    ("lon", float),
    ("height", float)
])

GeoCoord = namedtuple("GeoCoord", ["lat", "lon", "height"])

Geo2DCoord_dtype = np.dtype([
    ("lat", float),
    ("lon", float),
    ("cell", "U15")
])

PhaseCoord_dtype = np.dtype([
    ("longitude_asc", float),
    ("inclin", float),
    ("phase_on_orbit", float),
    ("height", float)
])

DecartCoord_dtype = np.dtype([
    ("x", float),
    ("y", float),
    ("z", float),
    ("cell", "U15")
])

@singleton
class CoordConverter:
    @staticmethod
    @staticmethod
    def phase_to_geo_np(phase_coord_np: npt.NDArray[PhaseCoord_dtype]) -> npt.NDArray[GeoCoord_dtype]:

        longitude_asc = phase_coord_np["longitude_asc"]
        inclin = phase_coord_np["inclin"]
        phase = phase_coord_np["phase_on_orbit"]
        heights = phase_coord_np["height"]

        lon_asc_rad = np.radians(longitude_asc)
        inclin_rad = np.radians(inclin)
        phase_rad = np.radians(phase)

        sin_lat = np.sin(phase_rad) * np.sin(inclin_rad)
        sin_lat = np.clip(sin_lat, -1.0, 1.0)
        lat_rad = np.arcsin(sin_lat)
        lat = np.degrees(lat_rad)

        lon = (
            longitude_asc +
            np.degrees(
                np.arctan2(
                    np.sin(phase_rad) * np.cos(inclin_rad),
                    np.cos(phase_rad)
                )
            )
        ) % 360

        result = np.empty(len(lat), dtype=GeoCoord_dtype)
        result["lat"] = lat
        result["lon"] = lon
        result["height"] = heights

        return result



    @staticmethod
    def geo_to_dec_np(geo_coord_np: npt.NDArray[GeoCoord_dtype_cell]) -> npt.NDArray[DecartCoord_dtype]:
        lats = geo_coord_np["lat"]
        lons = geo_coord_np["lon"]
        heights = geo_coord_np["height"]

        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)

        xs = (EARTH_RADIUS + heights) * np.cos(lats_rad) * np.cos(lons_rad)
        ys = (EARTH_RADIUS + heights) * np.cos(lats_rad) * np.sin(lons_rad)
        zs = (EARTH_RADIUS + heights) * np.sin(lats_rad)

        result = np.empty(len(lats), dtype=DecartCoord_dtype)
        result["x"] = xs
        result["y"] = ys
        result["z"] = zs

        result["cell"] = geo_coord_np["cell"]

        return result

    @staticmethod
    def geo_to_dec_np_sat(geo_coord_np: npt.NDArray[GeoCoord_dtype]) -> npt.NDArray[DecartCoord_dtype]:
        lats = geo_coord_np["lat"]
        lons = geo_coord_np["lon"]
        heights = geo_coord_np["height"]

        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)

        xs = (EARTH_RADIUS + heights) * np.cos(lats_rad) * np.cos(lons_rad)
        ys = (EARTH_RADIUS + heights) * np.cos(lats_rad) * np.sin(lons_rad)
        zs = (EARTH_RADIUS + heights) * np.sin(lats_rad)

        result = np.empty(len(lats), dtype=DecartCoord_dtype)
        result["x"] = xs
        result["y"] = ys
        result["z"] = zs

        return result

    @staticmethod
    def geo_to_dec_single(lat: float, lon: float, height: float) -> npt.NDArray[GeoCoord_dtype]:
        geo_coord_np = np.array([(lat, lon, height)], dtype=GeoCoord_dtype)
        return CoordConverter().geo_to_dec_np_sat(geo_coord_np)

    @staticmethod
    def geo_2d_to_dec_np(geo_2d_coord_np: npt.NDArray[Geo2DCoord_dtype]) -> npt.NDArray[DecartCoord_dtype]:
        geo_coord_np = np.empty(len(geo_2d_coord_np), dtype=GeoCoord_dtype_cell)
        geo_coord_np["lat"] = geo_2d_coord_np["lat"]
        geo_coord_np["lon"] = geo_2d_coord_np["lon"]
        geo_coord_np["cell"] = geo_2d_coord_np["cell"]
        geo_coord_np["height"] = 0.0
        return CoordConverter().geo_to_dec_np(geo_coord_np)