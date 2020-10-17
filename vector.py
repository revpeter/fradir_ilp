import math
from typing import List, Tuple

from numpy import arccos, array, cross, dot, pi
from numpy.linalg import det, norm


class Vector:
    __slots__ = ['x', 'y', 'z', 'id', 'label']

    def __init__(self, x: float, y: float, z: float, id: int = None, label: str = None):
        self.x = x
        self.y = y
        self.z = z
        self.id = id
        self.label = label

    # PROPERTIES #

    @property
    def xyz(self) -> Tuple[float]:
        return (self.x, self.y, self.z)

    @property
    def length(self) -> float:
        return math.sqrt(self.dot(self))

    @property
    def unit(self):
        return self.scale(1 / self.length)

    # GENERAL GEOMETRY METHODS #

    def dot(self, v) -> float:
        x = self.x * v.x
        y = self.y * v.y
        z = self.z * v.z
        return x + y + z

    def add(self, v):
        x = self.x + v.x
        y = self.y + v.y
        z = self.z + v.z
        return Vector(x, y, z)

    def div(self, v):
        return self.add(v.scale(-1))

    def vector(self, v):
        return v.add(self.scale(-1))

    def distance_to(self, v) -> float:
        return self.vector(v).length

    def scale(self, amount: float):
        x = self.x * amount
        y = self.y * amount
        z = self.z * amount
        return Vector(x, y, z)

    def shorten(self, w, amount: float):
        a = self.scale(1 - amount / 2.0)
        alfa = w.scale(amount / 2.0)
        b = w.scale(1 - amount / 2.0)
        beta = self.scale(amount / 2.0)
        return (a.add(alfa), b.add(beta))

    def cross(self, v):
        x = self.y * v.z - v.y * self.z
        y = self.z * v.x - v.z * self.x
        z = self.x * v.y - v.x * self.y
        return Vector(x, y, z)

    # FLAT MATHODS #

    def _distance_to_line_segment(self, v, w):
        # https://gist.github.com/nim65s/5e9902cd67f094ce65b0
        p = array([self.x, self.y])
        src = array([v.x, v.y])
        dst = array([w.x, w.y])
        if all(src == p) or all(dst == p):
            return 0
        if arccos(dot((p - src) / norm(p - src), (dst - src) / norm(dst - src))) > pi / 2:
            return norm(p - src)
        if arccos(dot((p - dst) / norm(p - dst), (src - dst) / norm(src - dst))) > pi / 2:
            return norm(p - dst)
        return norm(cross(src-dst, src-p))/norm(dst-src)

    # SPHERE METHODS #

    # length of shortest path between two points on sphere
    def _great_circle_distance_to(self, v):
        # https://en.wikipedia.org/wiki/N-vector
        dot_prod = self.dot(v)
        dot_prod = min(max(dot_prod, -1), 1)
        return math.acos(dot_prod)

    # three functions to determin the shortest path length between point and arc
    def _is_the_closest_point_on_great_circle_on_the_arc(self, v, w) -> float:
        # https://www.movable-type.co.uk/scripts/latlong-vectors.html#intersection
        n1 = v.cross(w)
        n2 = n1.cross(self)

        c1 = n1.cross(n2).unit
        c2 = n2.cross(n1).unit

        len_arc = v._great_circle_distance_to(w)
        len_c1_a = c1._great_circle_distance_to(v)
        len_c1_b = c1._great_circle_distance_to(w)
        len_c2_a = c2._great_circle_distance_to(v)
        len_c2_b = c2._great_circle_distance_to(w)

        guess_c1 = abs(len_c1_a + len_c1_b - len_arc)
        guess_c2 = abs(len_c2_a + len_c2_b - len_arc)

        return min(guess_c1, guess_c2) < 0.0001

    def _distance_to_great_circle_arc(self, v, w) -> float:
        if self._is_the_closest_point_on_great_circle_on_the_arc(v, w):
            # perpendicular_angular_distance_to_great_circle
            member1 = abs(self.dot(v.cross(w)))
            member2 = float(v.cross(w).length)
            return math.asin(member1 / member2)
        else:
            return min(self._great_circle_distance_to(v), self._great_circle_distance_to(w))
