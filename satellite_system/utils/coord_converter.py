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
    def phase_to_geo_np(phase_coord_np: npt.NDArray[PhaseCoord_dtype]) -> npt.NDArray[GeoCoord_dtype]:
        longitude_asceding_nodes = phase_coord_np["longitude_asc"]
        inclins = phase_coord_np["inclin"]
        phases_on_orbit = phase_coord_np["phase_on_orbit"]
        heights = phase_coord_np["height"]

        longitude_asceding_nodes_rad = np.radians(longitude_asceding_nodes)
        inclins_rad = np.radians(inclins)
        phases_on_orbit_rad = np.radians(phases_on_orbit)

        sin_lats = np.clip(
            np.sin(phases_on_orbit_rad) * np.sin(inclins_rad),
            -1.0, 1.0
        )

        lats_rad = np.arcsin(sin_lats)
        lats = np.degrees(lats_rad)

        def formula(longitude_asceding_nodes: np.ndarray, phases_on_orbit_rad: np.ndarray, inclins_rad: np.ndarray) -> np.ndarray:
            return (longitude_asceding_nodes + np.degrees(
                np.arctan2(
                    np.sin(phases_on_orbit_rad) * np.cos(inclins_rad),
                    np.cos(phases_on_orbit_rad)
                )
            )) % 360

        conditions = [phases_on_orbit_rad == 90, inclins_rad != 90, (0.5 * np.pi < phases_on_orbit_rad) & (phases_on_orbit_rad < 1.5 * np.pi), True]
        choices = [0, formula(longitude_asceding_nodes_rad, phases_on_orbit_rad, inclins_rad), (longitude_asceding_nodes + 180) % 360, longitude_asceding_nodes]
        lons = np.select(conditions, choices)

        result = np.empty(len(lats), dtype=GeoCoord_dtype)
        result["lat"] = lats
        result["lon"] = lons
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
        geo_coord_np["height"] = EARTH_RADIUS
        return CoordConverter().geo_to_dec_np(geo_coord_np)