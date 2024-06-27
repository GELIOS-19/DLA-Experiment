import copy
import enum
from typing import List, Self, Tuple, Optional, Dict
import random
import time
from collections import deque, defaultdict
import math
import cmath

import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

GLOBAL_DIRECTIONS = (
                (-1, -1),  # North West
                (0, -1),  # North
                (1, -1),  # North East
                (1, 0),  # East
                (1, 1),  # South East
                (0, 1),  # South
                (-1, 1),  # South West
                (-1, 0),  # West
)


class Direction(enum.Enum):
        NORTH_WEST = 0
        NORTH = 1
        NORTH_EAST = 2
        EAST = 3
        SOUTH_EAST = 4
        SOUTH = 5
        SOUTH_WEST = 6
        WEST = 7


def create_bresenham_line(initial_x: int, initial_y: int, final_x: int, final_y: int) -> List[Tuple[int, int]]:
        difference_x = abs(final_x - initial_x)
        difference_y = abs(final_y - initial_y)
        step_x = 1 if initial_x < final_x else -1
        step_y = 1 if initial_y < final_y else -1
        error = difference_x - difference_y
        line = []

        while True:
                line.append((initial_x, initial_y))

                if initial_x == final_x and initial_y == final_y:
                        break
                two_times_error = error * 2
                if two_times_error > -difference_y:
                        error -= difference_y
                        initial_x += step_x
                if two_times_error < difference_x:
                        error += difference_x
                        initial_y += step_y

        return line


class Boundary:
        points: list[list[int]]
        edges: list[list[list[int]]]
        size: int

        def __init__(self, points: list[list[int]]):
                if len(points) < 3:
                        raise ValueError("Boundary must have at least 3 points")

                self.points = points

        @property
        def edges(self) -> list[list[tuple[int, int]]]:
                edges = []

                for i in range(len(self.points)):
                        first_point = self.points[i]
                        second_point = self.points[i + 1] if i < len(self.points) - 1 else self.points[0]
                        line = create_bresenham_line(first_point[0], first_point[1], second_point[0], second_point[1])
                        edges.append(line)

                return edges

        @property
        def size(self) -> int:
                x_minimum = min(p[0] for p in self.points)
                x_maximum = max(p[0] for p in self.points)
                y_minimum = min(p[1] for p in self.points)
                y_maximum = max(p[1] for p in self.points)

                width = x_maximum - x_minimum
                height = y_maximum - y_minimum

                return max(width, height) + 1


class Pixel:
        x: int
        y: int
        boundary: bool
        flooded: bool
        frozen: bool
        weight: float
        struck: Optional[Self]
        coords: tuple[int, int]

        def __init__(self, x: int, y: int):
                self.x = x
                self.y = y
                self.boundary = False
                self.flooded = False
                self.frozen = False
                self.weight = 0
                self.struck = None

        @property
        def coords(self) -> tuple[int, int]:
                return self.x, self.y


class Image:
        boundary: Boundary
        size: int
        density: float
        grid: List[List[Pixel]]
        origin: Optional[Pixel]

        def __init__(self, boundary: Boundary):
                self.boundary = boundary
                self.size = self.boundary.size
                self.density = 0
                self.origin = None

                self.grid = []
                for x in range(self.size):
                        self.grid.append([])
                        for y in range(self.size):
                                self.grid[x].append(Pixel(x, y))

                queue = deque()

                # Initialize the queue with the boundary pixels
                for i in range(self.size):
                        if not self.grid[0][i].boundary:
                                queue.append((0, i))
                        if not self.grid[self.size - 1][i].boundary:
                                queue.append((self.size - 1, i))
                        if not self.grid[i][0].boundary:
                                queue.append((i, 0))
                        if not self.grid[i][self.size - 1].boundary:
                                queue.append((i, self.size - 1))

                # Perform the flood fill
                while queue:
                        x, y = queue.popleft()
                        if not self.grid[x][y].flooded and not self.grid[x][y].boundary:
                                self.grid[x][y].flooded = True
                                if x > 0:
                                        queue.append((x - 1, y))
                                if x < self.size - 1:
                                        queue.append((x + 1, y))
                                if y > 0:
                                        queue.append((x, y - 1))
                                if y < self.size - 1:
                                        queue.append((x, y + 1))

        def __getitem__(self, index: Tuple[int, int]) -> Pixel:
                x, y = index

                x = min(max(x, 0), self.size - 1)
                y = min(max(y, 0), self.size - 1)

                if not self.grid[y][x].flooded:
                        return self.grid[y][x]

                # BFS to find the closest boundary pixel
                queue = deque([(x, y)])
                visited = {x, y}

                while queue:
                        current_x, current_y = queue.popleft()
                        if self.grid[current_y][current_x].boundary:
                                return self.grid[current_y][current_x]

                        for neighbor_difference_x, neighbor_difference_y in GLOBAL_DIRECTIONS:
                                if 0 <= current_x + neighbor_difference_x < self.size and 0 <= current_y + neighbor_difference_y < self.size and (current_x + neighbor_difference_x, current_y + neighbor_difference_y) not in visited:
                                        visited.add((current_x + neighbor_difference_x, current_y + neighbor_difference_y))
                                        queue.append((current_x + neighbor_difference_x, current_y + neighbor_difference_y))

        def __add__(self, other: Self) -> Self:
                if self.size != other.size:
                        raise ArithmeticError("Cannot add the weights of two images with different sizes")

                if self.boundary.points != other.boundary.points:
                        raise ArithmeticError("Cannot add the weights of two images with different boundaries")

                new_image = Image(self.boundary)
                for x in range(self.size):
                        for y in range(self.size):
                                if not new_image.grid[x][y].flooded:
                                        new_image.grid[x][y].weight = self.grid[x][y].weight + other.grid[x][y].weight
                return new_image

        @property
        def weights(self) -> List[List[int | float]]:
                weights = [[]] * self.size
                for i in range(self.size):
                        weights[i] = [pixel.weight for pixel in self.grid[i]]
                return weights

        def is_traversable(self) -> bool:
                def is_in_bounds(x, y):
                        return 0 <= x < self.size and 0 <= y < self.size

                def breadth_first_search(start_pixel: Pixel, target_pixel: Pixel):
                        queue = [start_pixel]
                        visited = set()
                        visited.add((start_pixel.x, start_pixel.y))

                        while queue:
                                current_pixel = queue.pop(0)
                                if current_pixel == target_pixel:
                                        return True
                                for direction in GLOBAL_DIRECTIONS:
                                        nx, ny = (current_pixel.x + direction[0], current_pixel.y + direction[1])
                                        if is_in_bounds(nx, ny):
                                                neighbor = self.grid[nx][ny]
                                                if neighbor.frozen and (nx, ny) not in visited:
                                                        visited.add((nx, ny))
                                                        queue.append(neighbor)
                        return False

                if not self.origin or not self.origin.frozen:
                        return False

                frozen_pixels = [(x, y) for x in range(self.size) for y in range(self.size) if self.grid[x][y].frozen]

                if not frozen_pixels:
                        return False

                origin_pixel = self.origin
                for x, y in frozen_pixels:
                        if not breadth_first_search(self.grid[x][y], origin_pixel):
                                return False

                return True


def constrain(value: int | float, low: int | float, high: int | float) -> Tuple[int | float, bool]:
        return max(low, min(high, value)), value == max(low, min(high, value))


def bezier_sigmoid(length, slope, curve_point, precision=10000):
        def bezier_curve(time_value, control_point_0, control_point_1, control_point_2, control_point_3):
                return (1 - time_value)**3 * control_point_0[0] + 3 * (1 - time_value)**2 * time_value * control_point_1[0] + 3 * (1 - time_value) * time_value**2 * control_point_2[0] + time_value**3 * control_point_3[0], (1 - time_value)**3 * control_point_0[1] + 3 * (1 - time_value)**2 * time_value * control_point_1[1] + 3 * (1 - time_value) * time_value**2 * control_point_2[1] + time_value**3 * control_point_3[1]

        def vertical_line_test(points):
                x = 0
                for point in points:
                        if point[0] > x:
                                x = point[0]
                        elif point[0] < x:
                                return False
                return True

        control_point_0 = (0.0, 1.0)
        control_point_3 = (length, 0.0)

        control_line = control_point_0[1] - slope * control_point_0[0]
        initial_line_x = (1 - control_line) * slope
        final_line_x = (0 - control_line) * slope
        midpoint = (initial_line_x + final_line_x) / 2
        offset = curve_point - midpoint
        initial_line_x += offset
        final_line_x += offset

        control_point_1 = (initial_line_x, 1.0)
        control_point_2 = (final_line_x, 0.0)

        bezier_points = [bezier_curve(time_value / precision, control_point_0, control_point_1, control_point_2, control_point_3) for time_value in range(precision + 1)]

        if not vertical_line_test(bezier_points):
                raise ArithmeticError(f"For the given parameters (a={length}, m={slope}, b={curve_point}), a function cannot be formed.")

        return bezier_points


def smooth_falloff(x: int | float, k: int | float) -> float:
        return 1 - (1 / (1 + k * x))


def calculate_density(image: Image) -> float:
        frozen_pixels = sum(pixel.frozen for row in image.grid for pixel in row)
        total_pixels = image.size**2
        return frozen_pixels / total_pixels


def build_graph_adjacency_lists(traversable_image: Image) -> Tuple[List[List[Optional[int]]], List[List[Optional[int]]]]:
        inbound_adjacency_list: List[List[Optional[int]]] = [[] for _ in range(traversable_image.size**2)]
        for x in range(traversable_image.size):
                for y in range(traversable_image.size):
                        pixel = traversable_image.grid[x][y]
                        if pixel.frozen and pixel.struck:
                                edges_index = pixel.struck.x * traversable_image.size + pixel.struck.y
                                inbound_adjacency_list[edges_index].append(x * traversable_image.size + y)
                        elif not pixel.frozen:
                                inbound_adjacency_list[x * traversable_image.size + y].append(None)

        outbound_adjacency_list: List[List[Optional[int]]] = [[] for _ in range(traversable_image.size**2)]
        for edges_index, edges in enumerate(inbound_adjacency_list):
                for edge in edges:
                        if edge is not None:
                                outbound_adjacency_list[edge].append(edges_index)
                        else:
                                outbound_adjacency_list[edges_index].append(None)

        return inbound_adjacency_list, outbound_adjacency_list


def find_contiguous_line_segments(traversable_image: Image) -> Dict[int, Dict[str, int]]:
        origin = [traversable_image.origin.x, traversable_image.origin.y, 0]
        inbound_adjacency_list, _ = build_graph_adjacency_lists(traversable_image)

        visited = []
        stack = deque()

        visited.append(origin)
        stack.append(origin)

        mappings = defaultdict(lambda: dict())

        intersection_points = []
        count = 0
        current_direction = None

        while stack:
                subject = stack.pop()
                subject_index = subject[0] * traversable_image.size + subject[1]

                for node_index in reversed(inbound_adjacency_list[subject_index]):
                        node = [node_index // traversable_image.size, node_index % traversable_image.size, subject[2]]

                        previous_direction = current_direction

                        if node[0] == subject[0]:
                                if node[1] > subject[1]:
                                        current_direction = Direction.NORTH
                                elif node[1] < subject[1]:
                                        current_direction = Direction.SOUTH
                        elif node[1] == subject[1]:
                                if node[0] > subject[0]:
                                        current_direction = Direction.EAST
                                elif node[0] < subject[0]:
                                        current_direction = Direction.WEST
                        elif node[0] > subject[0]:
                                if node[1] > subject[1]:
                                        current_direction = Direction.NORTH_EAST
                                elif node[1] < subject[1]:
                                        current_direction = Direction.SOUTH_EAST
                        elif node[0] < subject[0]:
                                if node[1] > subject[1]:
                                        current_direction = Direction.NORTH_WEST
                                elif node[1] < subject[1]:
                                        current_direction = Direction.SOUTH_WEST

                        is_origin = subject == origin
                        is_intersection = len(inbound_adjacency_list[subject_index]) > 1
                        if is_intersection:
                                intersection_points.append(node)
                        is_direction_change_only = previous_direction is not None and (previous_direction != current_direction) and not is_intersection and subject not in intersection_points

                        if is_origin or is_intersection or is_direction_change_only:
                                mappings[count]["direction"] = current_direction
                                mappings[count]["starts_at"] = subject
                                node[2] = count
                                count += 1

                        if node[2] in mappings.keys():
                                mappings[node[2]]["ends_at"] = node

                        if DEBUG:
                                print(node[2], is_intersection, is_direction_change_only, previous_direction, current_direction, subject, "->", node)

                        visited.append(node)
                        stack.append(node)

        for key in mappings.keys():
                if len(mappings[key]["starts_at"]) == 3 and len(mappings[key]["ends_at"]) == 3:
                        mappings[key]["starts_at"].pop(2)
                        mappings[key]["ends_at"].pop(2)

        return dict(mappings)


def simulate_random_walk(image: Image, num_concurrent_walkers: int):
        # TODO: In the future, to make this code compatible with non-square geometry,
        #       we can use bresenham's line algorithm to roughly model an equation that
        #       follows the edges of this geometry. This edges tuple can be generated
        #       dependent on the geometry
        edges = image.boundary.edges

        walkers = []
        for walker in range(num_concurrent_walkers):
                edge = random.choice(edges)
                x, y = random.choice(edge)
                while image.grid[x][y].frozen:
                        x = random.randint(0, image.size - 1)
                        y = random.randint(0, image.size - 1)

                path = [(x, y)]

                walkers.append(path)

        while walkers:
                for i, path in enumerate(walkers):
                        direction = random.choice(GLOBAL_DIRECTIONS)

                        if image[path[-1][0] + direction[0], path[-1][1] + direction[1]].frozen:
                                previous_x = path[-1][0]
                                previous_y = path[-1][1]

                                pixel = image.grid[previous_x][previous_y]
                                pixel.struck = image[path[-1][0] + direction[0], path[-1][1] + direction[1]]
                                pixel.frozen = True
                                pixel.weight = 100
                                walkers.pop(i)

                        path.append(image[path[-1][0] + direction[0], path[-1][1] + direction[1]].coords)

        return walkers


def crisp_upscale(traversable_image: Image, new_image_size: int) -> Image:
        scale_factor = new_image_size / traversable_image.size
        new_image = Image(Boundary([[int(point[0] * scale_factor), int(point[1] * scale_factor)] for point in traversable_image.boundary.points]))

        new_image.origin = new_image.grid[int(traversable_image.origin.x * scale_factor)][int(traversable_image.origin.y * scale_factor)]

        for x in range(traversable_image.size):
                for y in range(traversable_image.size):
                        if traversable_image.grid[x][y].frozen:
                                core_pixel = new_image.grid[int(x * scale_factor)][int(y * scale_factor)]
                                core_pixel.frozen = True
                                core_pixel.weight = 200

        _, outbound_adjacency_list = build_graph_adjacency_lists(traversable_image)
        for edges_index, edges in enumerate(outbound_adjacency_list):
                if None not in edges:
                        initial_x = int((edges_index // traversable_image.size) * scale_factor)
                        initial_y = int((edges_index % traversable_image.size) * scale_factor)

                        for edge in edges:
                                final_x = int((edge // traversable_image.size) * scale_factor)
                                final_y = int((edge % traversable_image.size) * scale_factor)

                                line_points = create_bresenham_line(initial_x, initial_y, final_x, final_y)
                                for line_point_index, (line_point_x, line_point_y) in enumerate(line_points[:-1]):
                                        line_pixel = new_image.grid[line_point_x][line_point_y]
                                        line_pixel.weight = 100 + 20 * line_point_index
                                        line_pixel.frozen = True

                                        next_point = line_points[line_point_index + 1]
                                        line_pixel.struck = new_image.grid[next_point[0]][next_point[1]]

        new_image.density = calculate_density(new_image)

        if DEBUG:
                for x in range(new_image_size):
                        for y in range(new_image_size):
                                pixel = new_image.grid[x][y]
                                if pixel.struck is not None:
                                        print(f"({pixel.x}, {pixel.y}) -> ({pixel.struck.x}, {pixel.struck.y})")

        return new_image


def jitter_contiguous_lines(traversable_image: Image) -> Image:
        pass


def apply_downstream_height(traversable_image: Image, clamp: int) -> Image:
        new_image = Image(traversable_image.boundary)

        new_image.origin = new_image.grid[traversable_image.origin.x][traversable_image.origin.y]

        inbound_adjacency_list, _ = build_graph_adjacency_lists(traversable_image)

        def get_downstream_count(index: int) -> int:
                def depth_first_search(node, memo):
                        if node in memo:
                                return memo[node]
                        max_length = 0
                        for neighbor in inbound_adjacency_list[node]:
                                max_length = max(max_length, depth_first_search(neighbor, memo))
                        memo[node] = max_length + 1
                        return memo[node]

                return depth_first_search(index, {})

        downstream_counts = [0 for _ in range(traversable_image.size**2)]

        origin = (traversable_image.origin.x, traversable_image.origin.y)

        visited = [origin]
        queue = [origin]

        while queue:
                subject = queue.pop(0)
                subject_index = subject[0] * traversable_image.size + subject[1]

                downstream_counts[subject_index] = get_downstream_count(subject_index)

                for node in inbound_adjacency_list[subject_index]:
                        next_node = (node // traversable_image.size, node % traversable_image.size)
                        if next_node not in visited:
                                visited.append(next_node)
                                queue.append(next_node)

        for pixel_index, downstream_count in enumerate(downstream_counts):
                if downstream_count == 0:
                        continue

                x = pixel_index // traversable_image.size
                y = pixel_index % traversable_image.size
                pixel = new_image.grid[x][y]

                pixel.frozen = True
                pixel.weight = int(clamp * smooth_falloff(downstream_count, 0.05))

                if traversable_image.grid[x][y].struck:
                        pixel.struck = new_image.grid[traversable_image.grid[x][y].struck.x][traversable_image.grid[x][y].struck.y]

        new_image.density = calculate_density(new_image)

        return new_image


def bilinear_upscale(image: Image, new_image_size: int) -> Image:
        scale_factor = new_image_size / image.size
        new_image = Image(Boundary([[int(point[0] * scale_factor), int(point[1] * scale_factor)] for point in image.boundary.points]))

        for i in range(new_image_size):
                for j in range(new_image_size):
                        x = ((i + 0.5) / scale_factor) - 0.5
                        y = ((j + 0.5) / scale_factor) - 0.5

                        x_floor = int(x)
                        y_floor = int(y)

                        x_difference = x - x_floor
                        y_difference = y - y_floor

                        top_left_weight = image[x_floor, y_floor].weight
                        top_right_weight = image[constrain(x_floor + 1, 0, image.size - 1)[0], y_floor].weight
                        bottom_left_weight = image[x_floor, constrain(y_floor + 1, 0, image.size - 1)[0]].weight
                        bottom_right_weight = image[constrain(x_floor + 1, 0, image.size - 1)[0], constrain(y_floor + 1, 0, image.size - 1)[0]].weight

                        top_weight = top_right_weight * x_difference + top_left_weight * (1 - x_difference)
                        bottom_weight = bottom_right_weight * x_difference + bottom_left_weight * (1 - x_difference)

                        interpolated_weight = bottom_weight * y_difference + top_weight * (1 - y_difference)
                        new_image[i, j].weight = interpolated_weight

        return new_image


# BELOW FUNCTION WAS IMPLEMENTED BY CHATGPT
def bicubic_upscale(image: Image, new_image_size: int) -> Image:
        scale_factor = new_image_size / image.size
        new_image = Image(Boundary([[int(point[0] * scale_factor), int(point[1] * scale_factor)] for point in image.boundary.points]))

        def cubic(x):
                absolute_x = abs(x)
                if absolute_x <= 1:
                        return 1 - 2 * absolute_x**2 + absolute_x**3
                elif 1 < absolute_x < 2:
                        return 4 - 8 * absolute_x + 5 * absolute_x**2 - absolute_x**3
                else:
                        return 0

        def get_interpolated_value(x, y):
                x_floor = int(x)
                y_floor = int(y)
                x_difference = x - x_floor
                y_difference = y - y_floor

                result = 0
                for m in range(-1, 3):
                        for n in range(-1, 3):
                                weight = image[x_floor + m, y_floor + n].weight
                                result += weight * cubic(m - x_difference) * cubic(n - y_difference)

                return max(0, result)

        for i in range(new_image_size):
                for j in range(new_image_size):
                        original_x = (i + 0.5) / scale_factor - 0.5
                        original_y = (j + 0.5) / scale_factor - 0.5
                        interpolated_weight = get_interpolated_value(original_x, original_y)
                        new_image[i, j].weight = interpolated_weight

        return new_image


def gaussian_blur(image: Image, standard_deviation: int | float) -> Image:
        new_image = Image(image.boundary)

        def create_kernel(size, sigma):
                gaussian_filter = [[0] * size for _ in range(size)]

                center = size // 2
                constant_term = 1 / (2 * math.pi * sigma**2)

                total = 0
                for i in range(size):
                        for j in range(size):
                                x = i - center
                                y = j - center
                                gaussian_filter[i][j] = constant_term * math.e**-((x**2 + y**2) / (2 * sigma**2))
                                total += gaussian_filter[i][j]

                for i in range(size):
                        for j in range(size):
                                gaussian_filter[i][j] /= total

                return gaussian_filter

        kernel_base_size = round(6 * standard_deviation)
        kernel = create_kernel((kernel_base_size if kernel_base_size % 2 == 1 else kernel_base_size + 1), standard_deviation)

        def pad(matrix, pada_to_size):
                if not matrix:
                        return [[0] * pada_to_size for _ in range(pada_to_size)]

                rows = len(matrix)
                column = len(matrix[0])

                start_row = (pada_to_size - rows) // 2
                start_column = (pada_to_size - column) // 2

                padded_array = [[0] * pada_to_size for _ in range(pada_to_size)]

                for i in range(rows):
                        for j in range(column):
                                padded_array[start_row + i][start_column + j] = matrix[i][j]

                return padded_array

        def crop(matrix, original_rows, original_columns):
                size = len(matrix)

                start_row = (size - original_rows) // 2
                start_column = (size - original_columns) // 2

                cropped_array = [matrix[start_row + i][start_column:start_column + original_columns] for i in range(original_rows)]

                return cropped_array

        def nearest_power_of_two(n):
                return 1 << (n - 1).bit_length()

        def fast_fourier_transform(polynomial_coefficients):
                polynomial_length = len(polynomial_coefficients)
                if polynomial_length <= 1:
                        return polynomial_coefficients
                even_degrees = fast_fourier_transform(polynomial_coefficients[0::2])
                odd_degrees = fast_fourier_transform(polynomial_coefficients[1::2])
                twiddle_factors = [cmath.exp(-2j * cmath.pi * k / polynomial_length) * odd_degrees[k % len(odd_degrees)] for k in range(polynomial_length // 2)]
                return [even_degrees[k] + twiddle_factors[k] for k in range(polynomial_length // 2)] + [even_degrees[k] - twiddle_factors[k] for k in range(polynomial_length // 2)]

        def inverse_fast_fourier_transform(polynomial_values):
                polynomial_length = len(polynomial_values)
                if polynomial_length <= 1:
                        return polynomial_values
                even_degrees = inverse_fast_fourier_transform(polynomial_values[0::2])
                odd_degrees = inverse_fast_fourier_transform(polynomial_values[1::2])
                twiddle_factors = [cmath.exp(2j * cmath.pi * k / polynomial_length) * odd_degrees[k % len(odd_degrees)] for k in range(polynomial_length // 2)]
                return [(even_degrees[k] + twiddle_factors[k]) / 2 for k in range(polynomial_length // 2)] + [(even_degrees[k] - twiddle_factors[k]) / 2 for k in range(polynomial_length // 2)]

        def fast_fourier_transform_2d(matrix):
                fft_rows = [fast_fourier_transform(row) for row in matrix]
                transpose = list(zip(*fft_rows))
                fft_columns = [fast_fourier_transform(column) for column in transpose]
                return list(column for column in zip(*fft_columns))

        def inverse_fast_fourier_transform_2d(matrix):
                ifft_rows = [inverse_fast_fourier_transform(row) for row in matrix]
                transpose = list(zip(*ifft_rows))
                ifft_columns = [inverse_fast_fourier_transform(column) for column in transpose]
                return list(zip(*ifft_columns))

        def complex_multiplication(a, b):
                return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

        def fast_fourier_transform_shift(array):
                rows = len(array)
                columns = len(array[0])
                middle_row = rows // 2
                middle_column = columns // 2

                shifted_array = [[0] * columns for _ in range(rows)]

                for i in range(rows):
                        for j in range(columns):
                                new_i = (i + middle_row) % rows
                                new_j = (j + middle_column) % columns
                                shifted_array[new_i][new_j] = array[i][j]

                return shifted_array

        pad_to_size = nearest_power_of_two(max(len(image.weights), len(kernel)))
        padded_weights = pad(image.weights, pad_to_size)
        padded_kernel = fast_fourier_transform_shift(pad(kernel, pad_to_size))

        fft_weights = fast_fourier_transform_2d(padded_weights)
        fft_kernel = fast_fourier_transform_2d(padded_kernel)
        fft_product = complex_multiplication(fft_weights, fft_kernel)

        convolved_image = inverse_fast_fourier_transform_2d(fft_product)
        blurred_weights = crop(convolved_image, image.size, image.size)

        for i in range(new_image.size):
                for j in range(new_image.size):
                        new_image[i, j].weight = blurred_weights[i][j].real

        return new_image


def create_dla_noise(seed: int, initial_size: int, end_size: int, initial_density_threshold: float, density_falloff_extremity: float, density_falloff_bias: float, use_concurrent_walkers: bool, upscale_factor: float, jitter_range: int, smoothness: int) -> List[Image]:
        images = []

        image = Image(Boundary([[0, 0], [0, initial_size], [initial_size, 0], [initial_size, initial_size]]))
        image_sum = Image(Boundary([[0, 0], [0, initial_size], [initial_size, 0], [initial_size, initial_size]]))

        random.seed(seed)

        steps = 0
        current_size = initial_size
        while current_size * upscale_factor < end_size:
                current_size *= upscale_factor
                steps += 1

        print(f"Steps: {steps}")

        curve = bezier_sigmoid(steps, density_falloff_extremity, density_falloff_bias * steps)

        central_pixel = image.grid[image.size // 2][image.size // 2]
        central_pixel.weight = 255
        central_pixel.frozen = True

        image.origin = central_pixel
        image.density = calculate_density(image)

        for step in range(steps):
                step_density_threshold = 0
                for point in curve:
                        if abs(step - point[0]) < 0.001:
                                step_density_threshold = initial_density_threshold * point[1]

                if DEBUG:
                        print(f"Step: {step + 1}")
                        print(f"Image Size: {image.size}x{image.size}")
                        print(f"Image Density {image.density}")
                        print(f"Step Density Threshold: {step_density_threshold}")

                while image.density < step_density_threshold:
                        number_concurrent_walkers: int
                        if use_concurrent_walkers:
                                total_pixels = image.size**2
                                frozen_pixels = image.density * total_pixels
                                number_concurrent_walkers = max(1, int((step_density_threshold * total_pixels) - frozen_pixels))
                        else:
                                number_concurrent_walkers = 1

                        simulate_random_walk(image, number_concurrent_walkers)

                        image.density = calculate_density(image)

                images.append(copy.copy(image))

                image_sum += apply_downstream_height(image, 300)
                image_sum = gaussian_blur(bicubic_upscale(image_sum, int(image_sum.size * upscale_factor)), smoothness)
                image = crisp_upscale(image, int(image.size * upscale_factor))

        return images + [image_sum]


def display_image(image: Image) -> None:
        weights = []
        for x in range(image.size):
                weights.append([])
                for y in range(image.size):
                        weights[x].append(image.grid[y][x].weight)
        weights = np.array(weights)

        plt.imshow(weights, cmap="terrain", origin="lower")
        plt.colorbar(label="Weight")
        plt.title("Heightmap Based on 2D Array of Weights")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.show()


def main():
        start_time = time.time()

        images = create_dla_noise(seed=random.randint(0, 1000), initial_size=50, end_size=1000, initial_density_threshold=0.1, density_falloff_extremity=2, density_falloff_bias=1 / 3, use_concurrent_walkers=True, upscale_factor=2, jitter_range=5, smoothness=10)

        end_time = time.time()
        print(f"Image generation time: {end_time - start_time} seconds")

        if DEBUG:
                final_image = images[-1]
                display_image(final_image)


def test():
        image = Image(10)

        image.origin = image.grid[5][5]
        image.grid[5][5].frozen = True
        image.grid[5][5].weight = 100

        image.grid[6][4].frozen = True
        image.grid[6][4].weight = 50
        image.grid[6][4].struck = image.grid[5][5]

        # image.values[5][6].frozen = True
        # image.values[5][6].weight = 50
        # image.values[5][6].struck = image.values[5][5]

        image.grid[7][3].frozen = True
        image.grid[7][3].weight = 40
        image.grid[7][3].struck = image.grid[6][4]

        image.grid[5][3].frozen = True
        image.grid[5][3].weight = 40
        image.grid[5][3].struck = image.grid[6][4]

        image.grid[6][5].frozen = True
        image.grid[6][5].weight = 40
        image.grid[6][5].struck = image.grid[6][4]

        image.grid[7][5].frozen = True
        image.grid[7][5].weight = 30
        image.grid[7][5].struck = image.grid[6][5]

        # image.values[6][6].frozen = True
        # image.values[6][6].weight = 30
        # image.values[6][6].struck = image.values[6][5]


if __name__ == "__main__":
        main()
