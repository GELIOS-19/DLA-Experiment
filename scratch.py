from collections import deque
import math
from typing import *

import matplotlib.pyplot as plt
import numpy as np


class Boundary:
    points: list[list[int]]
    edges: list[list[list[int]]]
    size: int

    def __init__(self, points: list[list[int]]):
        if len(points) < 3:
            raise ValueError("Boundary must have at least 3 points")

        self.points = points

    @property
    def edges(
        self,
    ) -> list[list[tuple[int, int]]]:
        edges = []

        for i in range(len(self.points)):
            p0 = self.points[i]
            p1 = self.points[i + 1] if i < len(self.points) - 1 else self.points[0]
            line = self.get_line(p0[0], p0[1], p1[0], p1[1])
            edges.append(line)

        return edges

    @property
    def size(self) -> int:
        x_min = min(p[0] for p in self.points)
        x_max = max(p[0] for p in self.points)
        y_min = min(p[1] for p in self.points)
        y_max = max(p[1] for p in self.points)

        width = x_max - x_min
        height = y_max - y_min

        return max(width, height) + 1

    @staticmethod
    def get_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        line = []

        while True:
            line.append((x0, y0))

            if x0 == x1 and y0 == y1:
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x0 += sx
            if err2 < dx:
                err += dx
                y0 += sy

        return line


class Pixel:
    x: int
    y: int
    boundary: bool
    flooded: bool
    frozen: bool
    weight: float
    struck: Optional[Self]

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.boundary = False
        self.flooded = False
        self.frozen = False
        self.weight = 0
        self.struck = None


class Image:
    size: int
    grid: list[list[Pixel]]
    boundary: Boundary
    origin: Optional[Pixel]

    def __init__(self, boundary: Boundary):
        self.boundary = boundary
        self.size = boundary.size
        self.grid = [[Pixel(j, i) for j in range(self.size)] for i in range(self.size)]
        self.origin = None

        for edge in boundary.edges:
            for point in edge:
                self[point[0], point[1], False].boundary = True

        self.__flood_fill()

    def __flood_fill(self):
        queue = deque()

        # Initialize the queue with the boundary pixels
        for i in range(self.size):
            if not self[0, i, False].boundary:
                queue.append((0, i))
            if not self[self.size - 1, i, False].boundary:
                queue.append((self.size - 1, i))
            if not self[i, 0, False].boundary:
                queue.append((i, 0))
            if not self[i, self.size - 1, False].boundary:
                queue.append((i, self.size - 1))

        # Perform the flood fill
        while queue:
            x, y = queue.popleft()
            if not self[x, y, False].flooded and not self[x, y, False].boundary:
                self[x, y, False].flooded = True
                if x > 0:
                    queue.append((x - 1, y))
                if x < self.size - 1:
                    queue.append((x + 1, y))
                if y > 0:
                    queue.append((x, y - 1))
                if y < self.size - 1:
                    queue.append((x, y + 1))

    def weights(self) -> list[list[int]]:
        weights = [[]] * self.size
        for i in range(self.size):
            weights[i] = [pixel.weight for pixel in self.grid[i]]
        return weights

    def graph(self):
        pass

    def traversable(self) -> bool:
        pass

    def show(self) -> None:
        weights = []
        for i in range(self.size):
            weights.append([])
            for j in range(self.size):
                if self[j, i, False].flooded:
                    weights[i].append(-50)
                elif self[j, i, False].boundary:
                    weights[i].append(-10)
                else:
                    weights[i].append(self[j, i, False].weight)
        weights = np.array(weights)

        plt.imshow(
            weights,
            cmap="terrain",
            origin="upper",
        )
        plt.colorbar(label="Weight")
        plt.title("Heightmap Based on 2D Array of Weights")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.show()

    def __getitem__(self, index: tuple[int, int, bool]):
        x: int = index[0]
        y: int = index[1]
        clamp: bool = index[2]

        if not clamp:
            return self.grid[y][x]

        x = min(max(x, 0), self.size - 1)
        y = min(max(y, 0), self.size - 1)

        if not self.grid[y][x].flooded:
            return self.grid[y][x]

        # BFS to find the closest boundary pixel
        queue = deque([(x, y)])
        visited = {x, y}

        while queue:
            cx, cy = queue.popleft()
            if self.grid[cy][cx].boundary:
                return self.grid[cy][cx]

            for nx, ny in [
                (cx - 1, cy),
                (cx + 1, cy),
                (cx, cy - 1),
                (cx, cy + 1),
                (cx - 1, cy - 1),
                (cx + 1, cy + 1),
                (cx - 1, cy + 1),
                (cx + 1, cy - 1),
            ]:
                if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    def __add__(self, other: Self) -> Self:
        if self.size != other.size:
            raise ValueError("Image must be the same size")

        if self.boundary.points != other.boundary.points:
            raise ValueError("Image must have the same boundary")

        new_image = Image(Boundary(self.boundary.points))
        for i in range(self.size):
            for j in range(self.size):
                if not self.grid[i][j].flooded and other.grid[i][j].flooded:
                    new_image.grid[i][j].weight = self.grid[i][j].weight + other.grid[i][j].weight

        return new_image


def circle(radius: int, precision=5) -> list[list[int]]:
    points = []
    for i in range(precision):
        theta = 2 * math.pi * i / precision
        x = round(radius + radius * math.cos(theta))
        y = round(radius + radius * math.sin(theta))
        points.append([x, y])
    return points


if __name__ == "__main__":
    i1 = Image(Boundary([[0, 0], [10, 0], [5, 10]]))
    i2 = Image(Boundary([[0, 0], [10, 0], [5, 10]]))
    i3 = i1 + i2
    i3.show()
