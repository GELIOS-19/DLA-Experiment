import cmath
import copy
import enum
import math
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Self, Tuple

# External libraries only used for visualization
import matplotlib.pyplot as plt
import numpy as np

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


def perpendicular_direction(direction: Direction) -> Direction:
    match direction:
        case Direction.NORTH_WEST:
            return Direction.SOUTH_WEST
        case Direction.NORTH:
            return Direction.EAST
        case Direction.NORTH_EAST:
            return Direction.SOUTH_EAST
        case Direction.EAST:
            return Direction.SOUTH
        case Direction.SOUTH_EAST:
            return Direction.NORTH_EAST
        case Direction.SOUTH:
            return Direction.WEST
        case Direction.SOUTH_WEST:
            return Direction.NORTH_WEST
        case Direction.WEST:
            return Direction.NORTH


def bresenham_line(initial_x: int, initial_y: int, final_x: int, final_y: int) -> List[Tuple[int, int]]:
    x_difference = abs(final_x - initial_x)
    y_difference = abs(final_y - initial_y)
    x_step = 1 if initial_x < final_x else -1
    y_step = 1 if initial_y < final_y else -1
    error = x_difference - y_difference
    line_points = []

    while True:
        line_points.append((initial_x, initial_y))

        if initial_x == final_x and initial_y == final_y:
            break
        two_times_error = error * 2
        if two_times_error > -y_difference:
            error -= y_difference
            initial_x += x_step
        if two_times_error < x_difference:
            error += x_difference
            initial_y += y_step

    return line_points


def wu_line(initial_x: int, initial_y: int, final_x: int, final_y: int) -> List[Tuple[int, int]]:
    def plot(x, y):
        line_points.append((x, y))

    def ipart(x):
        return int(x)

    def round(x):
        return ipart(x + 0.5)

    def fpart(x):
        return x - ipart(x)

    def rfpart(x):
        return 1 - fpart(x)

    line_points = []

    steep = abs(final_y - initial_y) > abs(final_x - initial_x)

    if steep:
        initial_x, initial_y = initial_y, initial_x
        final_x, final_y = final_y, final_x

    if initial_x > final_x:
        initial_x, final_x = final_x, initial_x
        initial_y, final_y = final_y, initial_y

    dx = final_x - initial_x
    dy = final_y - initial_y

    gradient = dy / dx

    x_end = round(initial_x)
    y_end = initial_y + gradient * (x_end - initial_x)
    xgap = rfpart(initial_x + 0.5)
    x_pixel1 = x_end
    y_pixel1 = ipart(y_end)
    if steep:
        plot(y_pixel1, x_pixel1)
        plot(y_pixel1 + 1, x_pixel1)
    else:
        plot(x_pixel1, y_pixel1)
        plot(x_pixel1, y_pixel1 + 1)

    intery = y_end + gradient

    x_end = round(final_x)
    y_end = final_y + gradient * (x_end - final_x)
    xgap = fpart(final_x + 0.5)
    x_pixel2 = x_end
    y_pixel2 = ipart(y_end)
    if steep:
        plot(y_pixel2, x_pixel2)
        plot(y_pixel2 + 1, x_pixel2)
    else:
        plot(x_pixel2, y_pixel2)
        plot(x_pixel2, y_pixel2 + 1)

    if steep:
        for x in range(x_pixel1 + 1, x_pixel2):
            plot(ipart(intery), x)
            plot(ipart(intery) + 1, x)
            intery = intery + gradient
    else:
        for x in range(x_pixel1 + 1, x_pixel2):
            plot(x, ipart(intery))
            plot(x, ipart(intery) + 1)
            intery = intery + gradient

    return line_points


class Border:
    border_points: List[List[int]]
    edges: List[List[Tuple[int, int]]]
    size: int

    def __init__(self, border_points: List[List[int]]):
        border_points_without_duplicates = [i for n, i in enumerate(border_points) if i not in border_points[:n]]
        self.border_points = self.__clockwise_sort(border_points_without_duplicates)
        self.edges = []
        for i in range(len(self.border_points)):
            initial_point = self.border_points[i]
            final_point = self.border_points[i + 1] if i < len(self.border_points) - 1 else self.border_points[0]
            self.edges.append(bresenham_line(initial_point[0], initial_point[1], final_point[0], final_point[1]))

    @property
    def size(self) -> int:
        minimum_x = min(point[0] for point in self.border_points)
        maximum_x = max(point[0] for point in self.border_points)
        minimum_y = min(point[1] for point in self.border_points)
        maximum_y = max(point[1] for point in self.border_points)

        width = maximum_x - minimum_x
        height = maximum_y - minimum_y

        return max(width, height) + 1

    @staticmethod
    def __find_centroid(points):
        x_coordinates = [point[0] for point in points]
        y_coordinates = [point[1] for point in points]
        centroid = (sum(x_coordinates) / len(points), sum(y_coordinates) / len(points))
        return centroid

    @staticmethod
    def __polar_angle(point, centroid):
        angle = math.atan2(point[1] - centroid[1], point[0] - centroid[0])
        return angle

    @staticmethod
    def __distance(point, centroid):
        return (point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2

    @staticmethod
    def __clockwise_sort(points):
        centroid = Border.__find_centroid(points)
        points.sort(key=lambda point: (Border.__polar_angle(point, centroid), -Border.__distance(point, centroid)))
        return points

    @staticmethod
    def square(size):
        return Border([[0, 0], [size, 0], [0, size], [size, size]])

    @staticmethod
    def triangle(size):
        return Border([[0, 0], [size, 0], [size // 2, size]])

    @staticmethod
    def hexagon(size):
        return Border([[(size // 3), 0], [(size // 3) * 2, 0], [(size // 3), size], [(size // 3) * 2, size], [0, (size // 3)], [0, (size // 3) * 2], [size, (size // 3)], [size, (size // 3) * 2]])

    @staticmethod
    def circle(size):
        precision = 100
        radius = size // 2
        points = []
        for i in range(precision):
            theta = 2 * math.pi * i / precision
            x = round(radius + radius * math.cos(theta))
            y = round(radius + radius * math.sin(theta))
            points.append([x, y])
        return Border(points)

    def scale(self, new_size) -> Self:
        scale_factor = new_size / self.size
        return Border([[int(p[0] * scale_factor), int(p[1] * scale_factor)] for p in self.border_points])


class Pixel:
    x: int
    y: int
    border: bool
    dead: bool
    frozen: bool
    weight: int
    struck: Self
    coordinates: Tuple[int, int]

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.border = False
        self.dead = False
        self.frozen = False
        self.weight = 0
        self.struck = None

    @property
    def coordinates(self) -> Tuple[int, int]:
        return self.x, self.y

    def get_id(self, image_size: int) -> int:
        return self.x * image_size + self.y

    @staticmethod
    def get_coordinates_from_id(pixel_id: int, image_size: int) -> Tuple[int, int]:
        return pixel_id // image_size, pixel_id % image_size

    @property
    def representative_weight(self):
        if self.border:
            return -50
        elif self.dead:
            return -100
        else:
            return self.weight


class Image:
    border: Border
    size: int
    density: float
    grid: List[List[Pixel]]
    origin: Optional[Pixel]
    weights: List[List[int | float]]

    def __init__(self, border: Border):
        self.border = border
        self.size = border.size
        self.density = 0
        self.origin = None

        self.grid = []
        for x in range(self.size):
            self.grid.append([])
            for y in range(self.size):
                self.grid[x].append(Pixel(x, y))

        for edge in self.border.edges:
            for point in edge:
                self.grid[point[0]][point[1]].border = True

        queue = deque()

        # Initialize the queue with the boundary pixels
        for i in range(self.size):
            if not self.grid[0][i].border:
                queue.append((0, i))
            if not self.grid[self.size - 1][i].border:
                queue.append((self.size - 1, i))
            if not self.grid[i][0].border:
                queue.append((i, 0))
            if not self.grid[i][self.size - 1].border:
                queue.append((i, self.size - 1))

        # Perform the flood fill
        while queue:
            x, y = queue.popleft()
            if not self.grid[x][y].dead and not self.grid[x][y].border:
                self.grid[x][y].dead = True
                if x > 0:
                    queue.append((x - 1, y))
                if x < self.size - 1:
                    queue.append((x + 1, y))
                if y > 0:
                    queue.append((x, y - 1))
                if y < self.size - 1:
                    queue.append((x, y + 1))

    def __getitem__(self, index):
        x, y = index

        x = min(max(x, 0), self.size - 1)
        y = min(max(y, 0), self.size - 1)

        if not self.grid[x][y].dead:
            return self.grid[x][y]

        # BFS to find the closest boundary pixel
        queue = deque([(x, y)])
        visited = {x, y}

        while queue:
            current_x, current_y = queue.popleft()
            if not (self.grid[current_x][current_y].dead or self.grid[current_x][current_y].border):
                return self.grid[current_x][current_y]

            for neighbor_x, neighbor_y in [(current_x + direction[0], current_y + direction[1]) for direction in GLOBAL_DIRECTIONS]:
                if 0 <= neighbor_x < self.size and 0 <= neighbor_y < self.size and (neighbor_x, neighbor_y) not in visited:
                    visited.add((neighbor_x, neighbor_y))
                    queue.append((neighbor_x, neighbor_y))

    def __setitem__(self, index, value):
        pixel = self[index]
        pixel = value

    def __add__(self, other: Self) -> Self:
        new_image = Image(Border(self.border.border_points + other.border.border_points))
        for i in range(min(self.size, other.size)):
            for j in range(min(self.size, other.size)):
                new_image.grid[i][j].weight = self.grid[i][j].weight + other.grid[i][j].weight
        return new_image

    @property
    def raw_weights(self) -> List[List[int | float]]:
        raw_weights = []
        for i in range(self.size):
            raw_weights.append([])
            for j in range(self.size):
                raw_weights[i].append(self.grid[i][j].weight)
        return raw_weights

    @property
    def representative_weights(self):
        representative_weights = []
        for i in range(self.size):
            representative_weights.append([])
            for j in range(self.size):
                representative_weights[i].append(self.grid[j][i].representative_weight)
        return representative_weights

    def bounded_coordinates(self) -> List[Tuple[int, int]]:
        return [(x, y) for y in range(self.size) for x in range(self.size) if not (self.grid[x][y].dead or self.grid[x][y].border)]

    def graph(self):
        inbound_adjacency_list = [[] for _ in range(self.size ** 2)]
        outbound_adjacency_list = [[] for _ in range(self.size ** 2)]

        bounded_coordinates = self.bounded_coordinates()
        for x, y in bounded_coordinates:
            pixel = self[x, y]
            if pixel.frozen and pixel.struck:
                inbound_adjacency_list[pixel.struck.x * self.size + pixel.struck.y].append(x * self.size + y)
                outbound_adjacency_list[x * self.size + y].append(pixel.struck.x * self.size + pixel.struck.y)
            elif not pixel.frozen:
                inbound_adjacency_list[x * self.size + y].append(None)
                outbound_adjacency_list[x * self.size + y].append(None)

        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x][y].dead or self.grid[x][y].border:
                    inbound_adjacency_list[x * self.size + y].append(None)
                    outbound_adjacency_list[x * self.size + y].append(None)

        return inbound_adjacency_list, outbound_adjacency_list

    def traversable(self) -> bool:
        if self.origin is None:
            return False

        inbound_adjacency_list, outbound_adjacency_list = self.graph()
        leaves = []

        for index, (in_edges, out_edges) in enumerate(zip(inbound_adjacency_list, outbound_adjacency_list)):
            if len(in_edges) == 0 and len(out_edges) != 0:
                leaves.append(index)
                self[*Pixel.get_coordinates_from_id(index, self.size)].weight = 200

        reachable = []
        for leaf in leaves:
            dfs_visited = [leaf]
            dfs_stack = [leaf]

            dfs_depth = 1
            while dfs_stack:
                subject = dfs_stack.pop()

                if subject == self.origin.get_id(self.size):
                    reachable.append(True)

                self[*Pixel.get_coordinates_from_id(subject, self.size)].weight = dfs_depth + 10

                for node in outbound_adjacency_list[subject]:
                    if node not in dfs_visited:
                        dfs_visited.append(node)
                        dfs_stack.append(node)

                dfs_depth += 1

        return len(reachable) == len(leaves)


def clamp(value: int | float, low: int | float, high: int | float) -> int | float:
    return max(low, min(high, value))


def bezier_sigmoid(length, slope, curve_point, precision=10000):
    def bezier_curve(time_value, control_point_0, control_point_1, control_point_2, control_point_3):
        return (1 - time_value) ** 3 * control_point_0[0] + 3 * (1 - time_value) ** 2 * time_value * control_point_1[0] + 3 * (1 - time_value) * time_value**2 * control_point_2[0] + time_value**3 * control_point_3[0], (1 - time_value) ** 3 * control_point_0[1] + 3 * (1 - time_value) ** 2 * time_value * control_point_1[1] + 3 * (1 - time_value) * time_value**2 * control_point_2[1] + time_value**3 * control_point_3[1]

    def vertical_line_test(points):
        time_value = 0
        for point in points:
            if point[0] > time_value:
                time_value = point[0]
            elif point[0] < time_value:
                return False
        return True

    first_control_point = (0.0, 1.0)
    last_control_point = (length, 0.0)

    intersection_line = first_control_point[1] - slope * first_control_point[0]
    initial_line_x = (1 - intersection_line) * slope
    final_line_x = (0 - intersection_line) * slope
    midpoint = (initial_line_x + final_line_x) / 2
    offset = curve_point - midpoint
    initial_line_x += offset
    final_line_x += offset

    initial_intersection_line_control_point = (initial_line_x, 1.0)
    final_intersection_line_control_point = (final_line_x, 0.0)

    bezier_points = [bezier_curve(time_range / precision, first_control_point, initial_intersection_line_control_point, final_intersection_line_control_point, last_control_point) for time_range in range(precision + 1)]

    if not vertical_line_test(bezier_points):
        raise ArithmeticError(f"For the given parameters (a={length}, m={slope}, b={curve_point}), a function cannot be formed.")

    return bezier_points


def smooth_falloff(time_value: int | float, k: int | float) -> float:
    return 1 - (1 / (1 + k * time_value))


def shifted_exponential(time_value: int | float, k: int | float, a: int | float) -> float:
    return k ** (time_value - a)


def calculate_density(image: Image) -> float:
    bounded_coordinates = image.bounded_coordinates()
    frozen_pixels = sum(image[x, y].frozen for x, y in bounded_coordinates)
    total_pixels = len(bounded_coordinates)
    return frozen_pixels / total_pixels


def find_line_segments(image: Image) -> Dict[int, Dict[str, Direction | List[int]]]:
    origin = [image.origin.x, image.origin.y, 0]
    inbound_adjacency_list, _ = image.graph()

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
        subject_index = subject[0] * image.size + subject[1]

        for node_index in reversed(inbound_adjacency_list[subject_index]):
            node = [node_index // image.size, node_index % image.size, subject[2]]

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

            visited.append(node)
            stack.append(node)

    for key in mappings.keys():
        if len(mappings[key]["starts_at"]) == 3:
            mappings[key]["starts_at"].pop(2)
        if len(mappings[key]["ends_at"]) == 3:
            mappings[key]["ends_at"].pop(2)

    return dict(mappings)


def simulate_random_walk(image: Image, num_concurrent_walkers: int, walks_per_concurrent_walker: int):
    walkers = []
    for walker in range(num_concurrent_walkers):
        edge = random.choice(image.border.edges)
        x, y = random.choice(edge)
        bounded_coordinates = image.bounded_coordinates()
        while image.grid[x][y].frozen:
            x, y = random.choice(bounded_coordinates)

        path = [(x, y)]
        walkers.append(path)

    while walkers:
        for i, path in enumerate(walkers):
            for _ in range(walks_per_concurrent_walker):
                direction = random.choice(GLOBAL_DIRECTIONS)
                x, y = image[path[-1][0] + direction[0], path[-1][1] + direction[1]].coordinates

                if image.grid[x][y].frozen:
                    previous_x = path[-1][0]
                    previous_y = path[-1][1]

                    pixel: Pixel = image.grid[previous_x][previous_y]
                    pixel.struck = image.grid[x][y]
                    pixel.frozen = True
                    pixel.weight = 100
                    walkers.pop(i)
                    break

                path.append((x, y))

    return walkers


def crisp_upscale(image: Image, new_size_target: int) -> Image:
    new_image = Image(image.border.scale(new_size_target))
    scale_factor = new_image.size / image.size

    new_image.origin = new_image[int(image.origin.x * scale_factor), int(image.origin.y * scale_factor)]

    inbound_adjacency_list, _ = image.graph()

    bfs_visited = [(image.origin.get_id(image.size), None)]
    bfs_queue = [(image.origin.get_id(image.size), None)]

    while bfs_queue:
        subject, parent = bfs_queue.pop()

        x, y = Pixel.get_coordinates_from_id(subject, image.size)
        scaled_x = int(x * scale_factor)
        scaled_y = int(y * scale_factor)

        pixel = new_image[scaled_x, scaled_y]
        pixel.frozen = True
        pixel.weight = 1000

        if parent is not None:
            struck_x, struck_y = Pixel.get_coordinates_from_id(parent, image.size)
            scaled_struck_x = int(struck_x * scale_factor)
            scaled_struck_y = int(struck_y * scale_factor)

            line_points = bresenham_line(scaled_x, scaled_y, scaled_struck_x, scaled_struck_y)[1:-1]
            if len(line_points) > 0:
                for i, line_point in enumerate(line_points):
                    line_pixel = new_image[*line_point]
                    if not line_pixel.frozen:
                        line_pixel.frozen = True
                        line_pixel.weight = 1000

                        # Chain the struck property
                        if i < len(line_points) - 1:
                            line_pixel.struck = new_image[*line_points[i + 1]]
                        else:  # Connect the last gap pixel to the original pixel
                            line_pixel.struck = new_image[scaled_struck_x, scaled_struck_y]

                pixel.struck = new_image[*line_points[-1]]  # TODO: THIS LINE MAKES 0 SENSE!!!! WHY?????????
            else:
                pixel.struck = new_image[scaled_struck_x, scaled_struck_y]  # TODO: ????????

        for node in inbound_adjacency_list[subject]:
            if (node, subject) not in bfs_visited:
                bfs_visited.append((node, subject))
                bfs_queue.append((node, subject))

    new_image.density = calculate_density(new_image)
    return new_image


def jitter_lines(image: Image, jitter: int) -> Image:
    new_image = Image(Border(image.border.border_points))
    line_segments = find_line_segments(image)

    for segment_id, segment_info in line_segments.items():
        start_x, start_y = segment_info["starts_at"]
        end_x, end_y = segment_info["ends_at"]

        line_points = bresenham_line(start_x, start_y, end_x, end_y)

        jittered_points = [(line_points[0][0], line_points[0][1], image[*line_points[0]].weight)]
        for i in range(1, len(line_points) - 1):
            x, y = line_points[i]
            jitter_x = x + random.randint(-jitter, jitter)
            jitter_y = y + random.randint(-jitter, jitter)
            jittered_points.append((*image[jitter_x, jitter_y].coordinates, image[x, y].weight))
        jittered_points.append((line_points[-1][0], line_points[-1][1], image[*line_points[-1]].weight))

        for i in range(len(jittered_points) - 1):
            jitter_start_x, jitter_start_y, jitter_start_weight = jittered_points[i]
            jitter_end_x, jitter_end_y, jitter_end_weight = jittered_points[i + 1]
            jitter_line_points = bresenham_line(jitter_start_x, jitter_start_y, jitter_end_x, jitter_end_y)
            total_steps = len(jitter_line_points) - 1

            for j, jitter_line_point in enumerate(jitter_line_points):
                if total_steps > 0:
                    interpolation_factor = j / total_steps
                else:
                    interpolation_factor = 0
                new_weight = (1 - interpolation_factor) * jitter_start_weight + interpolation_factor * jitter_end_weight
                new_image[*jitter_line_point].weight = new_weight

    return new_image


def calculate_heights(image: Image, maximum_height: int | float, detail_falloff: float) -> Image:
    new_image = Image(Border(image.border.border_points))
    new_image.origin = new_image.grid[image.origin.x][image.origin.y]

    inbound_adjacency_list, outbound_adjacency_list = image.graph()

    def get_downstream_count(index):
        dfs_visited = [index]
        dfs_stack = [index]

        dfs_depth = 1
        while dfs_stack:
            subject = dfs_stack.pop()

            for node in inbound_adjacency_list[subject]:
                if node not in dfs_visited:
                    dfs_visited.append(node)
                    dfs_stack.append(node)

            dfs_depth += 1

        return dfs_depth

    bfs_visited = [new_image.origin.get_id(new_image.size)]
    bfs_queue = [new_image.origin.get_id(new_image.size)]

    while bfs_queue:
        subject = bfs_queue.pop(0)

        x, y = Pixel.get_coordinates_from_id(subject, new_image.size)
        new_image[x, y].frozen = True
        new_image[x, y].weight = maximum_height * smooth_falloff(get_downstream_count(subject), detail_falloff)
        if image[x, y].struck:
            new_image[x, y].struck = new_image[*image[x, y].struck.coordinates]

        for node in inbound_adjacency_list[subject]:
            if node not in bfs_visited:
                bfs_visited.append(node)
                bfs_queue.append(node)

    new_image.density = calculate_density(new_image)
    return new_image


def bilinear_upscale(image: Image, new_size_target: int) -> Image:
    new_image = Image(image.border.scale(new_size_target))
    scale_factor = new_image.size / image.size

    for i, j in new_image.bounded_coordinates():
        x = ((i + 0.5) / scale_factor) - 0.5
        y = ((j + 0.5) / scale_factor) - 0.5

        x_floor = int(x)
        y_floor = int(y)

        x_difference = x - x_floor
        y_difference = y - y_floor

        top_left_weight = image.grid[x_floor][y_floor].weight
        top_right_weight = image.grid[clamp(x_floor + 1, 0, image.size - 1)][y_floor].weight
        bottom_left_weight = image.grid[x_floor][clamp(y_floor + 1, 0, image.size - 1)].weight
        bottom_right_weight = image.grid[clamp(x_floor + 1, 0, image.size - 1)][clamp(y_floor + 1, 0, image.size - 1)].weight

        top_weight = top_right_weight * x_difference + top_left_weight * (1 - x_difference)
        bottom_weight = bottom_right_weight * x_difference + bottom_left_weight * (1 - x_difference)

        interpolated_weight = bottom_weight * y_difference + top_weight * (1 - y_difference)
        new_image.grid[i][j].weight = interpolated_weight

    return new_image


def bicubic_upscale(image: Image, new_size_target: int) -> Image:
    new_image = Image(image.border.scale(new_size_target))
    scale_factor = new_image.size / image.size

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
                x_index = int(clamp(x_floor + m, 0, image.size - 1))
                y_index = int(clamp(y_floor + n, 0, image.size - 1))
                weight = image.grid[x_index][y_index].weight
                result += weight * cubic(m - x_difference) * cubic(n - y_difference)

        return max(0, result)

    for i, j in new_image.bounded_coordinates():
        original_x = (i + 0.5) / scale_factor - 0.5
        original_y = (j + 0.5) / scale_factor - 0.5
        interpolated_weight = get_interpolated_value(original_x, original_y)
        new_image[i, j].weight = interpolated_weight

    return new_image


def gaussian_blur(image: Image, standard_deviation: int | float) -> Image:
    new_image = Image(Border(image.border.border_points))

    def create_kernel(size, sigma):
        gaussian_filter = [[0] * size for _ in range(size)]

        center = size // 2
        constant_term = 1 / (2 * math.pi * sigma**2)

        total = 0
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                gaussian_filter[i][j] = constant_term * math.e ** -((x**2 + y**2) / (2 * sigma**2))
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

        cropped_array = [matrix[start_row + i][start_column : start_column + original_columns] for i in range(original_rows)]

        return cropped_array

    def nearest_power_of_two(number):
        return 1 << (number - 1).bit_length()

    def fast_fourier_transform(polynomial_coefficients):
        polynomial_length = len(polynomial_coefficients)
        if polynomial_length <= 1:
            return polynomial_coefficients

        even_degree_terms = fast_fourier_transform(polynomial_coefficients[0::2])
        odd_degree_terms = fast_fourier_transform(polynomial_coefficients[1::2])

        twiddle_factors = [cmath.exp(-2j * cmath.pi * k / polynomial_length) * odd_degree_terms[k % len(odd_degree_terms)] for k in range(polynomial_length // 2)]
        return [even_degree_terms[k] + twiddle_factors[k] for k in range(polynomial_length // 2)] + [even_degree_terms[k] - twiddle_factors[k] for k in range(polynomial_length // 2)]

    def inverse_fast_fourier_transform(polynomial_values):
        polynomial_length = len(polynomial_values)
        if polynomial_length <= 1:
            return polynomial_values

        even_degree_terms = inverse_fast_fourier_transform(polynomial_values[0::2])
        odd_degree_terms = inverse_fast_fourier_transform(polynomial_values[1::2])

        twiddle_factors = [cmath.exp(2j * cmath.pi * k / polynomial_length) * odd_degree_terms[k % len(odd_degree_terms)] for k in range(polynomial_length // 2)]
        return [(even_degree_terms[k] + twiddle_factors[k]) / 2 for k in range(polynomial_length // 2)] + [(even_degree_terms[k] - twiddle_factors[k]) / 2 for k in range(polynomial_length // 2)]

    def fft_2d(matrix):
        fft_rows = [fast_fourier_transform(row) for row in matrix]
        transpose = list(zip(*fft_rows))
        fft_columns = [fast_fourier_transform(column) for column in transpose]
        return list(column for column in zip(*fft_columns))

    def ifft_2d(matrix):
        ifft_rows = [inverse_fast_fourier_transform(row) for row in matrix]
        transpose = list(zip(*ifft_rows))
        ifft_columns = [inverse_fast_fourier_transform(column) for column in transpose]
        return list(zip(*ifft_columns))

    def elementwise_matrix_multiplication(matrix_a, matrix_b):
        return [[matrix_a[i][j] * matrix_b[i][j] for j in range(len(matrix_a[0]))] for i in range(len(matrix_a))]

    def fft_shift(array):
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

    pad_to_size = nearest_power_of_two(max(len(image.raw_weights), len(kernel)))
    padded_weights = pad(image.raw_weights, pad_to_size)
    padded_kernel = fft_shift(pad(kernel, pad_to_size))

    fft_weights = fft_2d(padded_weights)
    fft_kernel = fft_2d(padded_kernel)
    fft_product = elementwise_matrix_multiplication(fft_weights, fft_kernel)

    convolved_image = ifft_2d(fft_product)
    blurred_weights = crop(convolved_image, image.size, image.size)

    for i, j in new_image.bounded_coordinates():
        new_image[i, j].weight = blurred_weights[i][j].real

    return new_image


def create_dla_noise(seed: int, initial_size: int, end_size: int, initial_density_threshold: float, density_falloff_extremity: float, density_falloff_bias: float, use_concurrent_walkers: bool, walks_per_concurrent_walker: int, upscale_factor: int, jitter_range: int, height_goal: float, detail_falloff: float, smoothness: int) -> List[Image]:
    images = []

    image = Image(Border.circle(initial_size))
    image_sum = Image(Border.circle(initial_size))

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
            print(f"Image Density: {image.density}")
            print(f"Step Density Threshold: {step_density_threshold}")

        while image.density < step_density_threshold:
            num_concurrent_walkers: int
            if use_concurrent_walkers:
                total_pixels = image.size**2
                frozen_pixels = image.density * total_pixels
                num_concurrent_walkers = max(1, int((step_density_threshold * total_pixels) - frozen_pixels))
            else:
                num_concurrent_walkers = 1

            if DEBUG:
                print(f"Step: {step + 1} :: Simulating Random Walk")
            simulate_random_walk(image, num_concurrent_walkers, walks_per_concurrent_walker)

            if DEBUG:
                print(f"Step: {step + 1} :: Calculating Density")
            image.density = calculate_density(image)

        images.append(copy.copy(image))

        if DEBUG:
            print(f"Step: {step + 1} :: Calculating Heights")
        image_with_height = calculate_heights(image, height_goal, detail_falloff)
        images.append(copy.copy(image_with_height))

        if DEBUG:
            print(f"Step: {step + 1} :: Jittering")
        image_with_jittered_lines = jitter_lines(image_with_height, jitter_range)
        images.append(copy.copy(image_with_jittered_lines))

        image_sum += image_with_jittered_lines

        if DEBUG:
            print(f"Step: {step + 1} :: Size Upscaling")
        image_sum = bicubic_upscale(image_sum, int(image_sum.size * upscale_factor))

        if DEBUG:
            print(f"Step: {step + 1} :: Applying Blur")
        image_sum = gaussian_blur(image_sum, smoothness)

        if DEBUG:
            print(f"Step: {step + 1} :: Crisp Upscaling")
        image = crisp_upscale(image, int(image.size * upscale_factor))

    return images + [image_sum]


def display_image(image: Image) -> None:
    weights = np.array(image.representative_weights)

    plt.imshow(weights, cmap="gray", origin="lower")
    plt.colorbar(label="Weight")
    plt.title("Heightmap Based on 2D Array of Weights")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.show()


def main():
    start_time = time.time()

    images = create_dla_noise(seed=random.randint(0, 1000), initial_size=100, end_size=1000, initial_density_threshold=0.1, density_falloff_extremity=2, density_falloff_bias=1 / 2, use_concurrent_walkers=True, walks_per_concurrent_walker=100, upscale_factor=2, jitter_range=10, height_goal=300, detail_falloff=0.005, smoothness=5)

    end_time = time.time()
    print(f"Image generation time: {end_time - start_time} seconds")

    if DEBUG:
        for image in images:
            display_image(image)


def test():
    # random.seed(0)
    image = Image(Border.circle(100))

    image[image.size // 2, image.size // 2].frozen = True
    image.origin = image.grid[image.size // 2][image.size // 2]
    while image.density < 0.1:
        simulate_random_walk(image, 1, 100)
        image.density = calculate_density(image)

    SCALE_FACTOR = 1.5

    print(image.traversable())
    u1 = crisp_upscale(image, int(image.size * SCALE_FACTOR))
    print(u1.traversable())
    u2 = calculate_heights(crisp_upscale(u1, int(u1.size * SCALE_FACTOR)), 300, 0)
    print(u2.traversable())
    u3 = crisp_upscale(u2, int(u2.size * SCALE_FACTOR))
    print(u3.traversable())
    u4 = crisp_upscale(u3, int(u3.size * SCALE_FACTOR))
    print(u4.traversable())

    display_image(image)
    display_image(u1)
    display_image(u2)
    display_image(u3)
    display_image(u4)
    display_image(calculate_heights(u4, 300, 1.015))


if __name__ == "__main__":
    main()
