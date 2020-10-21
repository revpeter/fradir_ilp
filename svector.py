import math
from typing import List, Tuple

from vector import Vector


class SVector:
    __slots__ = ['lat', 'lon', '_vector', '_scale']

    def __init__(self, lat: float, lon: float):
        self.lat, self.lon = lat, lon
        x, y, z = _xyz_from_latlon(lat, lon)
        self._vector = Vector(x, y, z)
        self._scale = math.cos(math.radians(42.))

    def __getattr__(self, attr):
        return getattr(self._vector, attr)

    # METHODS #

    def distance_to_section(self, src, dst) -> float:
        return self._distance_to_great_circle_arc(src, dst)

    def distance_to_point(self, point) -> float:
        return self._great_circle_distance_to(point)


def _xyz_from_latlon(lat: float, lon: float) -> Tuple[float]:
    φ, λ = math.radians(lat), math.radians(lon)
    x = math.cos(φ) * math.cos(λ)
    y = math.cos(φ) * math.sin(λ)
    z = math.sin(φ)
    return (x, y, z)
