"""Minimal GL2 model copy/subsurface geometry bookkeeping."""

from __future__ import annotations

import math


class GL2Model:
    def __init__(self, size, mesh, shaders, uniforms):
        self.width = size[0]
        self.height = size[1]
        self.mesh = mesh
        self.shaders = shaders
        self.uniforms = uniforms
        self.properties = None
        self.cached_texture = None
        self.forward = "IDENTITY"
        self.reverse = "IDENTITY"

    def copy(self):
        rv = GL2Model((self.width, self.height), self.mesh, self.shaders, self.uniforms)
        rv.forward = self.forward
        rv.reverse = self.reverse
        return rv

    def subsurface(self, rect):
        x, y, w, h = rect
        rv = self.copy()
        rv.width = int(math.ceil(w))
        rv.height = int(math.ceil(h))
        rv.reverse = ("offset", -x, -y, rv.reverse)
        rv.forward = ("offset", x, y, rv.forward)
        rv.mesh = ("crop", rv.mesh, (0, 0, w, h))
        return rv
