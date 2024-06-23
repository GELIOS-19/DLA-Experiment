import copy
import enum
from typing import List, Self, Tuple, Any, Optional, Dict
import random
import time
from collections import deque, defaultdict
import math  # Operations on complex numbers are supported in native python, but we must find a library for this in java
import cmath

import numpy as np  # Get rid of numpy
from numpy import ndarray, dtype
import matplotlib.pyplot as plt
"""
Goal:
        Based on a technique presented in
        https://www.youtube.com/watch?v=gsJHzBTPG0Y&t=715s We are attempting to
        generate heightmaps using a combination of DLA and blur algorithms.
        These heightmaps are suitable for generating mountains with natural
        looking veins.

Definitions:
        Image: An traversable_image that is represented by a square grid of floating point
                numbers between 0 and 255.
        Pixel: A grid location in the Image that holds a floating point value
                between 0 and 255. Additionally, a Pixel will keep track of the
                one pixel it connected to through a random walk. Pixels can
                perform random walks where the can go either up, down, left, or
                right.

Steps:
        General Algorithm:
                1. Starting with a small Image (see "Image" definition), we 
                        first place a pixel of maximum brightness at the center.
                2. At a pseudo-random grid location determined by a seed value, 
                        we place another pixel (see "Pixel" definition) and 
                        perform a random walk through the grid until the pixel 
                        is adjacent to an existing pixel in the grid. We freeze 
                        this pixel in place.
                3. We calculate the density of the traversable_image defined by comparing 
                        the number bright grid locations to the number of dark 
                        grid locations.
                4. We repeat steps 1-3 until the density passes a threshold, we
                        increase the size of the Image and upscale the pixels.

        Upscaling:
                There are two types of Image upscaling used in this algorithm: A
                crisp upscale and a blurry upscale.

                5. When we perform an upscale on the Image, we need to perform 
                        both a crisp upscale and a blurry upscale.
                6. Taking the Image with the crisp upscale, we continue to add
                        detail to this traversable_image through the process outlined in 
                        steps 1-4.
                7. Once the density desired in step 4 is achieved, we add the
                        detail from the crisp upscale to the blurry upscale, 
                        while keeping the crisp upscale for future crisp 
                        upscales.

        Crisp Upscale:
                8. When we perform step 2, we must keep track of the frozen
                        pixel that the new pixel gets stuck to.
                9. Using the connections from step 8, we can redraw these
                        connections as lines using the scale of the new Image.
                10. The lines from step 9 can be split at their  midpoint, and 
                        this midpoint can be jittered to create a  more natural
                        result.

        Blurry Upscale:
                11. Using the original crisp traversable_image, we first assign the
                        outermost pixels a weight of 1.
                12. We then recursively assign other pixels the maximum
                        weight of all the pixels downstream from the target.
                13. Use the smooth falloff formula (1 - (1/(1+h)) to clamp
                        the weights of pixels of higher weights.
                14. We then use bicubic interpolation to upscale the 
                        Image to the new size.
                15. Lastly, we use convolutions to assign each pixel a
                        weighted average of the adjacent pixels.
"""

# TODO: Rewrite with a class based approach

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


class Pixel:
        x: int
        y: int
        weight: int
        struck: Self
        frozen: bool

        def __init__(self, x: int, y: int):
                self.x = x
                self.y = y
                self.weight = 0
                self.frozen = False
                self.struck = None

        def __str__(self):
                return f"Pixel(x={self.x}, y={self.y}, weight={self.weight}, frozen={self.frozen}, struck=({self.struck.x if self.struck else None}, {self.struck.y if self.struck else None}))"


class Image:
        size: int
        density: float
        grid: List[List[Pixel]]
        origin: Optional[Pixel]

        def __init__(self, size: int):
                self.size = size
                self.density = 0
                self.origin = None

                self.grid = []
                for x in range(size):
                        self.grid.append([])
                        for y in range(size):
                                self.grid[x].append(Pixel(x, y))

        def __getitem__(self, index):
                x, y = index
                if 0 <= x < self.size and 0 <= y < self.size:
                        return self.grid[x][y]
                else:
                        raise IndexError("Pixel index out of range")

        def __add__(self, other: Self) -> Self:
                if self.size != other.size:
                        raise ArithmeticError("Cannot add the weights of two images with different sizes")
                new_image = Image(self.size)
                for i in range(self.size):
                        for j in range(self.size):
                                new_image[i, j].weight = self[i, j].weight + other[i, j].weight
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

                # Find all frozen pixels
                frozen_pixels = [(x, y) for x in range(self.size) for y in range(self.size) if self.grid[x][y].frozen]

                if not frozen_pixels:
                        return False

                # Check if all frozen pixels can reach the origin
                origin_pixel = self.origin
                for x, y in frozen_pixels:
                        if not breadth_first_search(self.grid[x][y], origin_pixel):
                                return False

                return True


def constrain(value: int | float, low: int | float, high: int | float) -> Tuple[int | float, bool]:
        return max(low, min(high, value)), value == max(low, min(high, value))


# TODO: Port this function to pure python
def bezier_sigmoid(a: int | float, m: int | float, b: int | float, precision=10000) -> ndarray[Any, dtype[Any]] | None:

        def bezier_curve(time_value, control_point_0, control_point_1, control_point_2, control_point_3):
                return (1 - time_value)**3 * control_point_0 + 3 * (1 - time_value)**2 * time_value * control_point_1 + 3 * (1 - time_value) * time_value**2 * control_point_2 + time_value**3 * control_point_3

        def vertical_line_test(bezier_points):
                x = 0
                for point in bezier_points:
                        if point[0] > x:
                                x = point[0]
                        if point[0] < x:
                                return False
                return True

        # Define control points
        control_point_0 = np.array([0.0, 1.0])
        control_point_3 = np.array([a, 0.0])

        control_line = control_point_0[1] - m * control_point_0[0]
        initial_line_x = (1 - control_line) / m
        final_line_x = (0 - control_line) / m
        midpoint = (initial_line_x + final_line_x) / 2
        offset = b - midpoint
        initial_line_x += offset
        final_line_x += offset

        control_point_1 = np.array([initial_line_x, 1.0])
        control_point_2 = np.array([final_line_x, 0.0])

        # Find the bezier points
        time_values = np.linspace(0, 1, precision)
        bezier_points = np.array([bezier_curve(t, control_point_0, control_point_1, control_point_2, control_point_3) for t in time_values])

        # Check if curve passes the vertical line test
        is_function = vertical_line_test(bezier_points)
        if not is_function:
                raise ArithmeticError(f"For the given parameters (a={a}, m={m}, b={b}), a function cannot be formed.")

        return bezier_points


def smooth_falloff(x: int | float, k: int | float) -> float:
        return 1 - (1 / (1 + k * x))


def calculate_density(image: Image) -> float:
        frozen_pixels = sum(pixel.frozen for row in image.grid for pixel in row)
        total_pixels = image.size**2
        return frozen_pixels / total_pixels


# TODO: Optimize this function
def build_graph_adjacency_lists(traversable_image: Image) -> Tuple[List[List[Optional[int]]], List[List[Optional[int]]]]:
        inbound_edges: List[List[int | None]] = [[] for _ in range(traversable_image.size**2)]
        for x in range(traversable_image.size):
                for y in range(traversable_image.size):
                        pixel = traversable_image.grid[x][y]
                        if pixel.frozen and pixel.struck:
                                i = pixel.struck.x * traversable_image.size + pixel.struck.y
                                inbound_edges[i].append(x * traversable_image.size + y)
                        elif not pixel.frozen:
                                inbound_edges[x * traversable_image.size + y].append(None)

        outbound_edges: List[List[int | None]] = [[] for _ in range(traversable_image.size**2)]
        for i, inbound_connections in enumerate(inbound_edges):
                for inbound_connection in inbound_connections:
                        if inbound_connection is not None:
                                outbound_edges[inbound_connection].append(i)
                        else:
                                outbound_edges[i].append(None)

        return inbound_edges, outbound_edges


def draw_bresenham_line(initial_x: int, initial_y: int, final_x: int, final_y: int) -> List[Tuple[int, int]]:
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


def find_contiguous_line_segments(traversable_image: Image) -> Any:
        # Use DFS to find the coordinates of every line segment
        # We must use DFS to traverse the entire graph, inbound and outbound
        # connections are given by the get_connections function
        # When traversing with DFS, if there are multiple paths to explore at
        # a node, then that node represents the end of the line segment we just
        # traversed and the beginning of all the line segments that may form
        # from each path. Think about this recursively.
        # The line_segments list should contain pairs of the endpoints of each
        # line segment found
        origin = [traversable_image.origin.x, traversable_image.origin.y, 0]
        inbound, _ = build_graph_adjacency_lists(traversable_image)

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

                for node_index in reversed(inbound[subject_index]):
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
                        is_intersection = len(inbound[subject_index]) > 1
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

        # Delete the extra segment number in the starts_at and ends_at fields
        for key in mappings.keys():
                if len(mappings[key]["starts_at"]) == 3 and len(mappings[key]["ends_at"]) == 3:
                        mappings[key]["starts_at"].pop(2)
                        mappings[key]["ends_at"].pop(2)

        return dict(mappings)


def simulate_random_walk(image: Image, num_concurrent_walkers: int):
        # Define edges and directions
        # TODO: In the future, to make this code compatible with non-square geometry,
        #       we can use bresenham's line algorithm to roughly model an equation that
        #       follows the edges of this geometry. This edges tuple can be generated
        #       dependent on the geometry
        edges = (
                        ((0, image.size - 1), (0, 0)),  # Top
                        ((0, image.size - 1), (image.size - 1, image.size - 1)),  # Bottom
                        ((image.size - 1, image.size - 1), (0, image.size - 1)),  # Right
                        ((0, 0), (0, image.size - 1)),  # Left
        )

        # Create a list of walkers
        walkers = []
        for walker in range(num_concurrent_walkers):
                # Place a pixel at a random position along an edge of the
                # traversable_image
                edge = random.choice(edges)
                x = random.randint(edge[0][0], edge[0][1])
                y = random.randint(edge[1][0], edge[1][1])
                while image.grid[x][y].frozen:
                        x = random.randint(0, image.size - 1)
                        y = random.randint(0, image.size - 1)

                path = [(x, y)]

                walkers.append(path)

        # Perform concurrent random walks
        while walkers:
                for i, path in enumerate(walkers):
                        # Randomly choose a direction to walk in
                        direction = random.choice(GLOBAL_DIRECTIONS)
                        x, _ = constrain(path[-1][0] + direction[0], 0, image.size - 1)
                        y, _ = constrain(path[-1][1] + direction[1], 0, image.size - 1)

                        # Once we find the coordinates of a frozen pixel, we will freeze the previous pixel along the path of the random walk
                        if image.grid[x][y].frozen:
                                previous_x = path[-1][0]
                                previous_y = path[-1][1]

                                pixel = image.grid[previous_x][previous_y]
                                pixel.struck = image.grid[x][y]
                                pixel.frozen = True
                                pixel.weight = 100
                                walkers.pop(i)

                        path.append((x, y))

        return walkers


def crisp_upscale(traversable_image: Image, new_image_size: int) -> Image:
        # Create a new traversable_image
        new_image = Image(new_image_size)
        scale_factor = new_image_size / traversable_image.size

        # Preserve the traversable_image origin
        new_image.origin = new_image.grid[int(traversable_image.origin.x * scale_factor)][int(traversable_image.origin.y * scale_factor)]

        # Translate pixels from old traversable_image onto new traversable_image
        for x in range(traversable_image.size):
                for y in range(traversable_image.size):
                        if traversable_image.grid[x][y].frozen:
                                core_pixel = new_image.grid[int(x * scale_factor)][int(y * scale_factor)]
                                core_pixel.frozen = True
                                core_pixel.weight = 200

        # Reconstruct outbound connections
        _, outbound = build_graph_adjacency_lists(traversable_image)
        for connections_index, connections in enumerate(outbound):
                if None not in connections:
                        initial_x = int((connections_index // traversable_image.size) * scale_factor)
                        initial_y = int((connections_index % traversable_image.size) * scale_factor)

                        for connection in connections:
                                final_x = int((connection // traversable_image.size) * scale_factor)
                                final_y = int((connection % traversable_image.size) * scale_factor)

                                line_points = draw_bresenham_line(initial_x, initial_y, final_x, final_y)
                                for line_point_index, (line_point_x, line_point_y) in enumerate(line_points[:-1]):
                                        line_pixel = new_image.grid[line_point_x][line_point_y]
                                        line_pixel.weight = 100 + 20 * line_point_index
                                        line_pixel.frozen = True

                                        next_point = line_points[line_point_index + 1]
                                        line_pixel.struck = new_image.grid[next_point[0]][next_point[1]]

        # Calculate new traversable_image density
        new_image.density = calculate_density(new_image)

        if DEBUG:
                for x in range(new_image_size):
                        for y in range(new_image_size):
                                pixel = new_image.grid[x][y]
                                if pixel.struck is not None:
                                        print(f"({pixel.x}, {pixel.y}) -> ({pixel.struck.x}, {pixel.struck.y})")

        return new_image


def jitter_contiguous_lines(traversable_image: Image) -> Image:
        # TODO: 2 things to experiment with:
        #       - Using bezier curves to jitter line segments
        pass


def apply_downstream_height(traversable_image: Image, clamp: int) -> Image:
        new_image = Image(traversable_image.size)

        # Set the origin of the new traversable_image to the same as it was in the input
        new_image.origin = new_image.grid[traversable_image.origin.x][traversable_image.origin.y]

        inbound, _ = build_graph_adjacency_lists(traversable_image)

        def get_downstream_count(index: int) -> int:
                # Get the maximum number of downstream pixels for any pixel
                # A downstream pixel is a pixel contained in inbound[index]
                # For any index, there will be multiple downstream paths
                # We must find the length of the longest path and return this
                # number plus one to account for the fact that blank pixels
                # should have a weight of 0

                # DFS to find the longest path of downstream nodes
                def dfs(node, memo):
                        if node in memo:
                                return memo[node]
                        max_length = 0
                        for neighbor in inbound[node]:
                                max_length = max(max_length, dfs(neighbor, memo))
                        memo[node] = max_length + 1
                        return memo[node]

                return dfs(index, {})

        # Use the downstream count of each pixel to calculate its brightness
        downstream_counts = [0 for _ in range(traversable_image.size**2)]

        # Start at the origin of the traversable_image and use BFS to explore the graph
        origin = (traversable_image.origin.x, traversable_image.origin.y)

        visited = [origin]
        queue = [origin]

        while queue:
                subject = queue.pop(0)
                subject_index = subject[0] * traversable_image.size + subject[1]

                downstream_counts[subject_index] = get_downstream_count(subject_index)

                for node in inbound[subject_index]:
                        next_node = (node // traversable_image.size, node % traversable_image.size)
                        if next_node not in visited:
                                visited.append(next_node)
                                queue.append(next_node)

        # Using the downstream_counts list, we can redraw the traversable_image where the weight is the downstream count of each pixel
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


# TODO: Implement BICUBIC for better results
def bilinear_upscale(image: Image, new_image_size: int) -> Image:
        new_image = Image(new_image_size)
        scale_factor = new_image_size / image.size

        for i in range(new_image_size):
                for j in range(new_image_size):
                        x = ((i + 0.5) / scale_factor) - 0.5
                        y = ((j + 0.5) / scale_factor) - 0.5

                        x_floor = int(x)
                        y_floor = int(y)

                        x_difference = (x - x_floor)
                        y_difference = (y - y_floor)

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
        new_image = Image(new_image_size)
        scale_factor = new_image_size / image.size

        # Cubic interpolation function
        def cubic(x):
                absolute_x = abs(x)
                if absolute_x <= 1:
                        return 1 - 2 * absolute_x**2 + absolute_x**3
                elif 1 < absolute_x < 2:
                        return 4 - 8 * absolute_x + 5 * absolute_x**2 - absolute_x**3
                else:
                        return 0

        # Get interpolated value at specified coordinate
        def get_interpolated_value(x, y):
                x_floor = int(x)
                y_floor = int(y)
                x_difference = x - x_floor
                y_difference = y - y_floor

                result = 0
                for m in range(-1, 3):
                        for n in range(-1, 3):
                                x_index = int(constrain(x_floor + m, 0, image.size - 1)[0])
                                y_index = int(constrain(y_floor + n, 0, image.size - 1)[0])
                                weight = image[x_index, y_index].weight
                                result += weight * cubic(m - x_difference) * cubic(n - y_difference)

                return max(0, result)

        # Apply interpolation across the new image
        for i in range(new_image_size):
                for j in range(new_image_size):
                        original_x = (i + 0.5) / scale_factor - 0.5
                        original_y = (j + 0.5) / scale_factor - 0.5
                        interpolated_weight = get_interpolated_value(original_x, original_y)
                        new_image[i, j].weight = interpolated_weight

        return new_image


def gaussian_blur(image: Image, standard_deviation: int | float) -> Image:
        # We need to perform gaussian blurs on the image using convolutions
        # A convolution is a mathematical operation that may be performed on
        # two multidimensional data, similar to multiplication and addition
        # In the context of 1D data such as arrays, convolutions can be
        # described as reversing the second array, offsetting it by some value,
        # and summing the multiplication of overlapping terms
        #
        # a1: [1, 2, 3, 4, 5 ]
        # a2: [6, 7, 8, 9, 10]
        #
        # Reverse the second array and offset it
        # a1:          [ 1, 2, 3, 4, 5]
        # a2: [10, 9, 8, 7, 6]
        # Now, sum the product of overlapping terms
        # Result for shown step of the convolution: 1*7 + 2*6
        #
        # The fast_fourier_transform (Fast Fourier Transform) algorithm can efficiently convert
        # between coefficient and value representations of polynomial functions
        # fast_fourier_transform is additionally used to convert a function from spatial domain to
        # frequency domain
        # By converting a m of the image weights and the convolution kernel
        # into frequency domain, we effectively create two polynomials
        # Multiplying these polynomials will result in the convolution of the
        # kernel over the image weights
        # By using inverse_fast_fourier_transform, we can convert the product polynomial from frequency
        # domain back into spatial domain, returning the weights of the blurred
        # image

        new_image = Image(image.size)

        # Get the gaussian filter kernel for convolution
        def create_kernel(size, sigma):
                gaussian_filter = [[0] * size for _ in range(size)]

                center = size // 2
                c = 1 / (2 * math.pi * sigma**2)

                total = 0
                for i in range(size):
                        for j in range(size):
                                x = i - center
                                y = j - center
                                gaussian_filter[i][j] = c * math.e**-((x**2 + y**2) / (2 * sigma**2))
                                total += gaussian_filter[i][j]

                for i in range(size):
                        for j in range(size):
                                gaussian_filter[i][j] /= total

                return gaussian_filter

        kernel_base_size = round(6 * standard_deviation)
        kernel = create_kernel(kernel_base_size if kernel_base_size % 2 == 1 else kernel_base_size + 1, standard_deviation)

        # Perform convolution using fast_fourier_transform
        def pad(matrix, pada_to_size):
                if not matrix:
                        return [[0] * pada_to_size for _ in range(pada_to_size)]

                # Get the current dimensions of the array
                rows = len(matrix)
                column = len(matrix[0])

                # Calculate the starting indices to center the original array
                start_row = (pada_to_size - rows) // 2
                start_column = (pada_to_size - column) // 2

                # Create a new array of the desired size filled with zeros
                padded_array = [[0] * pada_to_size for _ in range(pada_to_size)]

                # Copy the original array into the new array centered
                for i in range(rows):
                        for j in range(column):
                                padded_array[start_row + i][start_column + j] = matrix[i][j]

                return padded_array

        def crop(matrix, original_rows, original_columns):
                # Get the current dimensions of the padded array
                size = len(matrix)

                # Calculate the starting indices to get the original array
                start_row = (size - original_rows) // 2
                start_column = (size - original_columns) // 2

                # Extract the original array
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
                fast_fourier_transform_rows = [fast_fourier_transform(row) for row in matrix]
                transpose = list(zip(*fast_fourier_transform_rows))
                fast_fourier_transform_columns = [fast_fourier_transform(column) for column in transpose]
                return list(column for column in zip(*fast_fourier_transform_columns))

        def inverse_fast_fourier_transform_2d(matrix):
                inverse_fast_fourier_transform_rows = [inverse_fast_fourier_transform(row) for row in matrix]
                transpose = list(zip(*inverse_fast_fourier_transform_rows))
                inverse_fast_fourier_transform_columns = [inverse_fast_fourier_transform(column) for column in transpose]
                return list(zip(*inverse_fast_fourier_transform_columns))

        def complex_multiplication(a, b):
                return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

        def fast_fourier_transform_shift(array):
                rows = len(array)
                columns = len(array[0])
                middle_row = rows // 2
                middle_column = columns // 2

                # Create a new array with the same size
                shifted_array = [[0] * columns for _ in range(rows)]

                for i in range(rows):
                        for j in range(columns):
                                new_i = (i + middle_row) % rows
                                new_j = (j + middle_column) % columns
                                shifted_array[new_i][new_j] = array[i][j]

                return shifted_array

        # Pad the image weights and the kernel
        pad_to_size = nearest_power_of_two(max(len(image.weights), len(kernel)))
        padded_weights = pad(image.weights, pad_to_size)
        padded_kernel = fast_fourier_transform_shift(pad(kernel, pad_to_size))  # We must account for the shift of the 0-frequency element in the padded kernel

        # Apply fast_fourier_transform_2d to the weights and the kernel to convert them to frequency domain
        fast_fourier_transform_weights = fast_fourier_transform_2d(padded_weights)
        fast_fourier_transform_kernel = fast_fourier_transform_2d(padded_kernel)

        # Calculate the product
        fast_fourier_transform_product = complex_multiplication(fast_fourier_transform_weights, fast_fourier_transform_kernel)

        # Use inverse_fast_fourier_transform_2d to convert the product back into spatial domain
        convolved_image = inverse_fast_fourier_transform_2d(fast_fourier_transform_product)

        # Crop image to original size
        blurred_weights = crop(convolved_image, image.size, image.size)

        # Transfer blurred_weights to new image
        for i in range(new_image.size):
                for j in range(new_image.size):
                        new_image[i, j].weight = blurred_weights[i][j].real

        return new_image


def create_dla_noise(seed: int, initial_size: int, end_size: int, initial_density_threshold: float, density_falloff_extremity: float, density_falloff_bias: float, use_concurrent_walkers: bool, upscale_factor: float, jitter_range: int) -> List[Image]:
        images = []

        image = Image(initial_size)

        # Seed the random generator
        random.seed(seed)

        # Calculate the number of steps needed to reach end_size
        # Each iteration must progressively add less density in order for the algorithm to remain efficient
        steps = 0
        current_size = initial_size
        while current_size < end_size:
                current_size *= upscale_factor
                steps += 1

        print(f"Steps: {steps}")

        # Compute the falloff curve given the number of steps
        curve = bezier_sigmoid(steps, density_falloff_extremity, density_falloff_bias * steps)

        # First we must set the central pixel to have maximum weight
        central_pixel = image.grid[image.size // 2][image.size // 2]
        central_pixel.weight = 255
        central_pixel.frozen = True

        image.origin = central_pixel
        image.density = calculate_density(image)

        for step in range(steps):
                # Calculate the density threshold for this step
                step_density_threshold = 0
                for point in curve:
                        if abs(step - point[0]) < 0.001:
                                step_density_threshold = initial_density_threshold * point[1]

                if DEBUG:
                        print(f"Step: {step + 1}")
                        print(f"Image Size: {image.size}x{image.size}")
                        print(f"Image Density {image.density}")
                        print(f"Step Density Threshold: {step_density_threshold}")

                # Simulate random walks
                while image.density < step_density_threshold:
                        # The number of concurrent walkers is calculated based on density estimation
                        number_concurrent_walkers: int
                        if use_concurrent_walkers:
                                total_pixels = image.size**2
                                frozen_pixels = image.density * total_pixels
                                number_concurrent_walkers = max(1, int((step_density_threshold * total_pixels) - frozen_pixels))
                        else:
                                number_concurrent_walkers = 1

                        simulate_random_walk(image, number_concurrent_walkers)

                        # Update traversable_image density
                        image.density = calculate_density(image)

                # Add traversable_image to images
                images.append(copy.copy(image))

                # Upscale the traversable_image once step_density_threshold is met
                image = crisp_upscale(image, int(image.size * upscale_factor))

        return images


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

        images = create_dla_noise(seed=random.randint(0, 1000), initial_size=50, end_size=1000, initial_density_threshold=0.1, density_falloff_extremity=2, density_falloff_bias=1 / 2, use_concurrent_walkers=True, upscale_factor=2, jitter_range=5)

        end_time = time.time()
        print(f"Image generation time: {end_time - start_time} seconds")

        if DEBUG:
                image = images[0]
                blurry_image = gaussian_blur(bicubic_upscale(apply_downstream_height(image, 300), 500), 10)
                display_image(blurry_image)


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

        # blurred_image = gaussian_blur(image, image.size // 2)
        # display_image(blurred_image)

        display_image(image)
        upscaled = bicubic_upscale(image, 300)
        display_image(upscaled)
        display_image(gaussian_blur(upscaled, 10))


if __name__ == "__main__":
        test()
