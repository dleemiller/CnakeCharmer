import numpy as np


class BaselineEncoder:
    def __init__(self, height_factor, thicknesses):
        self.height_factor = float(height_factor)
        self.thicknesses = int(thicknesses)

    def plot(self, y_true, x, y, height):
        if x >= y_true.shape[1] or y >= y_true.shape[0] or x < 0 or y < 0:
            return
        y_true[y][x][0] = 1.0
        y_true[y][x][1] = height * self.height_factor

    def plot_radius(self, y_true, x, y, height):
        radius = self.thicknesses
        if radius == 0:
            radius = height // 2
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                self.plot(y_true, x + i, y + j, height)

    def draw_vertical_line(self, y_true, line):
        x0, y0, x1, y1, height = line
        for yy in range(y0, y1):
            self.plot_radius(y_true, x0, yy, height)

    def draw_line(self, y_true, line):
        x0, y0, x1, y1, height = line
        deltax = x1 - x0
        deltay = y1 - y0
        if deltax == 0:
            return self.draw_vertical_line(y_true, line)

        deltaerr = abs(float(deltay) / float(deltax))
        error = 0.0
        y = y0

        for x in range(x0, x1):
            self.plot_radius(y_true, x, y, height)
            error += deltaerr
            while error >= 0.5:
                sign = -1 if deltay < 0 else 1
                y += sign
                error -= 1.0

    def encode(self, image_size, base_lines):
        y_true = np.zeros((image_size[1], image_size[0], 2), dtype=np.float32)
        for i in range(base_lines.shape[0]):
            self.draw_line(y_true, base_lines[i])
        return y_true
