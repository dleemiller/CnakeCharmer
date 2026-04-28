import math
import random


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def unit(v):
    n = math.sqrt(dot(v, v))
    if n == 0.0:
        raise ValueError("zero-length vector")
    inv = 1.0 / n
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def random_in_unit_disk():
    while True:
        p = (2.0 * random.random() - 1.0, 2.0 * random.random() - 1.0, 0.0)
        if dot(p, p) < 1.0:
            return p


class Camera:
    def __init__(self, look_from, look_at, vup, vfov, aspect_ratio, aperture, focus_dist):
        theta = vfov * math.pi / 180.0
        half_height = math.tan(theta / 2.0)
        half_width = aspect_ratio * half_height

        self.lens_radius = aperture / 2.0
        self.w = unit(v_sub(look_from, look_at))
        self.u = unit(cross(vup, self.w))
        self.v = cross(self.w, self.u)
        self.origin = look_from

        self.lower_left_corner = v_sub(
            v_sub(
                v_sub(self.origin, v_mul(self.u, half_width * focus_dist)),
                v_mul(self.v, half_height * focus_dist),
            ),
            v_mul(self.w, focus_dist),
        )
        self.horizontal = v_mul(self.u, 2.0 * half_width * focus_dist)
        self.vertical = v_mul(self.v, 2.0 * half_height * focus_dist)

    def get_ray(self, s, t):
        rd = v_mul(random_in_unit_disk(), self.lens_radius)
        offset = v_add(v_mul(self.u, rd[0]), v_mul(self.v, rd[1]))
        direction = v_sub(
            v_add(
                v_add(self.lower_left_corner, v_mul(self.horizontal, s)), v_mul(self.vertical, t)
            ),
            v_add(self.origin, offset),
        )
        return v_add(self.origin, offset), direction
