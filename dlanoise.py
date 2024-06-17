import copy
from typing import List, Self, Tuple, Any
import random
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, dtype

"""
Goal:
    Based on a technique presented in
    https://www.youtube.com/watch?v=gsJHzBTPG0Y&t=715s We are attempting to
    generate heightmaps using a combination of DLA and blur algorithms.
    These heightmaps are suitable for generating mountains with natural
    looking veins.

Definitions:
    Image: An image that is represented by a square grid of floating point
            numbers between 0 and 255.
    Pixel: A grid location in the Image that holds a floating point value
            between 0 and 255. Additionally, a Pixel will keep track of the
            one pixel it connected to through a random walk. Pixels can
            perform random walks where the can go either up, down, left, or
            right.

Steps:
    General Algorithm:
        1. Starting with a small Image (see "Image" definition), we first
            place a pixel of maximum brightness at the center.
        2. At a pseudo-random grid location determined by a seed value, we
            place another pixel (see "Pixel" definition) and perform a
            random walk through the grid until the pixel is adjacent to an
            existing pixel in the grid. We freeze this pixel in place.
        3. We calculate the density of the image defined by comparing the
            number bright grid locations to the number of dark grid
            locations.
        4. We repeat steps 1-3 until the density passes a threshold, we
            increase the size of the Image and upscale the pixels.

    Upscaling:
        There are two types of Image upscaling used in this algorithm: A
        crisp upscale and a blurry upscale.

        5. When we perform an upscale on the Image, we need to perform both
            a crisp upscale and a blurry upscale.
        6. Taking the Image with the crisp upscale, we continue to add
            detail to this image through the process outlined in steps 1-4.
        7. Once the density desired in step 4 is achieved, we add the
            detail from the crisp upscale to the blurry upscale, while
            keeping the crisp upscale for future crisp upscales.

        Crisp Upscale:
           8. When we perform step 2, we must keep track of the frozen
               pixel that the new pixel gets stuck to.
           9. Using the connections from step 8, we can redraw these
               connections as lines using the scale of the new Image.
           10. The lines from step 9 can be split at their midpoint,
               and this midpoint can be jittered to create a more
               natural result.

        Blurry Upscale:
            11. Using the original crisp image, we first assign the
                outermost pixels a weight of 1.
            12. We then recursively assign other pixels the maximum
                weight of all the pixels downstream from the target.
            13. Use the smooth falloff formula (1 - (1/(1+h)) to clamp
                the weights of pixels of higher weights.
            14. We then use spherical linear interpolation to upscale the Image
                to the new size.
            15. Lastly, we use convolutions to assign each pixel a
                weighted average of the adjacent pixels.
"""

DEBUG = True


class Pixel:
    x: int
    y: int
    weight: int
    stuck_with: Self
    frozen: bool

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.weight = 0
        self.frozen = False
        self.stuck_with = None


class Image:
    size: int
    density: float
    values: List[List[Pixel]]
    origin: Pixel | None

    def __init__(self, size: int):
        self.size = size
        self.density = 0
        self.origin = None

        self.values = []
        for x in range(size):
            self.values.append([])
            for y in range(size):
                self.values[x].append(Pixel(x, y))


def constrain(value: int | float, low: int | float, high: int | float) -> Tuple[int | float, bool]:
    return max(low, min(high, value)), value == max(low, min(high, value))


def falloff_curve(a: int | float, m: int | float, b: int | float, precision=10000) -> ndarray[Any, dtype[Any]] | None:
    def bezier_curve(t, P0, P1, P2, P3):
        return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3

    def vertical_line_test(bezier_points):
        x_progression = 0
        for point in bezier_points:
            if point[0] > x_progression:
                x_progression = point[0]
            if point[0] < x_progression:
                return False
        return True

    # Define control points
    P0 = np.array([0.0, 1.0])
    P3 = np.array([a, 0.0])

    # Calculate the y-intercept of the line (c)
    c = P0[1] - m * P0[0]

    # Calculate x1 for P1 at y = 1
    x1 = (1 - c) / m

    # Calculate x2 for P2 at y = 0
    x2 = (0 - c) / m

    # Adjust x1 and x2 so that their midpoint is b
    midpoint = (x1 + x2) / 2
    offset = b - midpoint
    x1 += offset
    x2 += offset
    P1 = np.array([x1, 1.0])
    P2 = np.array([x2, 0.0])

    # Create t values
    t_values = np.linspace(0, 1, precision)

    # Calculate Bézier curve points
    bezier_points = np.array([bezier_curve(t, P0, P1, P2, P3) for t in t_values])

    # Check if curve passes the vertical line test
    is_function = vertical_line_test(bezier_points)
    if not is_function:
        raise ArithmeticError(f"For the given parameters (a={a}, m={m}, b={b}), a function cannot be formed.")

    return bezier_points


def calculate_density(image: Image) -> float:
    frozen_pixels = sum(pixel.frozen for row in image.values for pixel in row)
    total_pixels = image.size**2
    return frozen_pixels / total_pixels


# TODO: Optimize this function
def get_connections(image: Image) -> Tuple[List[List[int | None]], List[List[int | None]]]:
    inbound: List[List[int | None]] = [[] for _ in range(image.size**2)]
    for x in range(image.size):
        for y in range(image.size):
            pixel = image.values[x][y]
            if pixel.frozen and pixel.stuck_with:
                index = pixel.stuck_with.x * image.size + pixel.stuck_with.y
                inbound[index].append(x * image.size + y)
            elif not pixel.frozen:
                inbound[x * image.size + y].append(None)

    outbound: List[List[int | None]] = [[] for _ in range(image.size**2)]
    for index, inbound_connections in enumerate(inbound):
        for inbound_connection in inbound_connections:
            if inbound_connection is not None:
                outbound[inbound_connection].append(index)
            else:
                outbound[index].append(None)  # TODO: Figure out why this makes sense

    return inbound, outbound


def draw_bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
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
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return line


def find_straight_line_segments(image: Image) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    line_segments = []
    visited = set()

    for x in range(image.size):
        for y in range(image.size):
            if (x, y) in visited or not image.values[x][y].frozen:
                continue  # Skip visited or non-frozen pixels

            # Check for horizontal segments
            start_x = x
            end_x = x
            while end_x + 1 < image.size and image.values[end_x + 1][y].frozen:
                end_x += 1
                visited.add((end_x, y))

            if end_x > start_x:  # Only count segments longer than 1 pixel
                line_segments.append(((start_x, y), (end_x, y)))

            # Check for vertical segments
            start_y = y
            end_y = y
            while end_y + 1 < image.size and image.values[x][end_y + 1].frozen:
                end_y += 1
                visited.add((x, end_y))

            if end_y > start_y:
                line_segments.append(((x, start_y), (x, end_y)))

    # Find split points for line segment
    split_line_segments = copy.deepcopy(line_segments)
    for i, line_segment in enumerate(line_segments):
        (x0, y0), (x1, y1) = line_segment
        line_points = draw_bresenham(x0, y0, x1, y1)
        split_points = set()

        for line_point in line_points:
            x, y = line_point

            neighbor_points = ()
            next_points = ()
            if y0 == y1:
                # We have a horizontal line, we need to check if it is split by
                # any vertical lines
                neighbor_points = (
                    (0, -1),  # North
                    (0, 1),  # South
                )
                next_points = (
                    (1, 0),  # West
                    (1, 0),  # East
                )
            elif x0 == x1:
                # We have a vertical line, we need to check if it is split by any
                # horizontal lines
                neighbor_points = (
                    (-1, 0),  # West
                    (1, 0),  # East
                )
                next_points = (
                    (0, -1),  # North
                    (0, 1),  # South
                )

            for dx, dy in neighbor_points:
                neighbor_point_x, _ = constrain(x + dx, 0, image.size)
                neighbor_point_y, _ = constrain(y + dy, 0, image.size)

                for nx, ny in next_points:
                    next_point_x, _ = constrain(x + nx, 0, image.size)
                    next_point_y, _ = constrain(y + ny, 0, image.size)

                    if image.values[next_point_x][next_point_y].frozen:
                        if image.values[neighbor_point_x][neighbor_point_y].frozen:
                            split_points.add((x, y))

        # Now lets split the line segment according to the split points
        def split_segments(target, segments, split_points):
            target_segment = segments[target]
            new_segments = []
            e0 = target_segment[0]
            for i in range(len(split_points) + 1):
                if i == len(split_points):
                    e1 = target_segment[1]
                else:
                    e1 = split_points[i]
                new_segments.append((e0, e1))
                e0 = e1
            return new_segments

        if split_points:
            new_segments = split_segments(i, line_segments, list(split_points))
            # New segments must be inserted at i in place of what is already
            # there
            real_i = split_line_segments.index(line_segments[i])
            split_line_segments = split_line_segments[:real_i] + new_segments + split_line_segments[real_i + 1:]

    return split_line_segments

def simulate_random_walk(image: Image, num_concurrent_walkers: int):
    # Define edges and directions
    # TODO: In the future, to make this code compatible with non-square geometry,
    # we can use bresenham's line algorithm to roughly model an equation that
    # follows the edges of this geometry. This edges tuple can be generated
    # dependent on the geometry
    # TODO: IDT Using edges is worth it
    edges = (
        ((0, image.size - 1), (0, 0)),  # Top
        ((0, image.size - 1), (image.size - 1, image.size - 1)),  # Bottom
        ((image.size - 1, image.size - 1), (0, image.size - 1)),  # Right
        ((0, 0), (0, image.size - 1)),  # Left
    )

    directions = (
        (0, -1),  # North
        # (1, -1),  # North East
        (1, 0),  # East
        # (1, 1),  # South East
        (0, 1),  # South
        # (-1, 1),  # South West
        (-1, 0),  # West
        # (-1, -1),  # North West
    )

    # Create a list of walkers
    walkers = []
    for walker in range(num_concurrent_walkers):
        # Place a pixel at a random position along an edge of the image
        # edge = random.choice(edges)
        # x = random.randint(edge[0][0], edge[0][1])
        # y = random.randint(edge[1][0], edge[1][1])
        x = random.randint(0, image.size - 1)
        y = random.randint(0, image.size - 1)
        while image.values[x][y].frozen:
            x = random.randint(0, image.size - 1)
            y = random.randint(0, image.size - 1)

        path = [(x, y)]

        walkers.append(path)

    # Perform concurrent random walks
    while walkers:
        for i, path in enumerate(walkers):
            # Randomly choose a direction to walk in
            direction = random.choice(directions)
            x, _ = constrain(path[-1][0] + direction[0], 0, image.size - 1)
            y, _ = constrain(path[-1][1] + direction[1], 0, image.size - 1)

            # Once we find the coordinates of a frozen pixel, we will freeze
            # the previous pixel along the path of the random walk
            if image.values[x][y].frozen:
                prev_x = path[-1][0]
                prev_y = path[-1][1]

                pixel = image.values[prev_x][prev_y]
                pixel.stuck_with = image.values[x][y]
                pixel.frozen = True
                pixel.weight = 100
                walkers.pop(i)

            path.append((x, y))

    return walkers


def crisp_upscale(image: Image, new_image_size: int, midpoint_jitter: int) -> Image:
    # Create a new image
    new_image = Image(new_image_size)

    # Translate pixels from old image onto new image
    scale_factor = new_image_size / image.size

    # Preserve the image origin
    new_image.origin = new_image.values[int(image.origin.x * scale_factor)][int(image.origin.y * scale_factor)]

    for x in range(image.size):
        for y in range(image.size):
            if image.values[x][y].frozen:
                pixel = image.values[x][y]
                new_pixel = new_image.values[int(x * scale_factor)][int(y * scale_factor)]
                new_pixel.frozen = True
                new_pixel.weight = pixel.weight

    # Reconstruct outbound connections
    inbound, outbound = get_connections(image)
    for i, outbound_connections in enumerate(outbound):
        if None not in outbound_connections:
            x0 = int((i // image.size) * scale_factor)
            y0 = int((i % image.size) * scale_factor)
            for outbound_connection in outbound_connections:
                x1 = int((outbound_connection // image.size) * scale_factor)
                y1 = int((outbound_connection % image.size) * scale_factor)

                bresenham_line = draw_bresenham(x0, y0, x1, y1)
                for line_coord_i, (lx, ly) in enumerate(bresenham_line[:-1]):
                    line_pixel = new_image.values[lx][ly]
                    line_pixel.weight = 100 + 20 * line_coord_i
                    line_pixel.frozen = True
                    line_pixel.stuck_with = new_image.values[bresenham_line[line_coord_i + 1][0]][bresenham_line[line_coord_i + 1][1]]

    if DEBUG:
        for x in range(new_image_size):
            for y in range(new_image_size):
                pixel = new_image.values[x][y]
                if pixel.stuck_with is not None:
                    print(f"({pixel.x}, {pixel.y}) -> ({pixel.stuck_with.x}, {pixel.stuck_with.y})")

    new_image.density = calculate_density(new_image)

    return new_image


def blurry_upscale(image: Image) -> Image:
    pass


def perform_dla(
    seed: int,
    initial_size: int,
    end_size: int,
    initial_density_threshold: float,
    density_falloff_extremity: float,
    density_falloff_bias: float,
    use_concurrent_walkers: bool,
    upscale_factor: float,
    upscale_jitter: int,
) -> List[Image]:
    images = []
    image = Image(initial_size)

    # Seed the random generator
    random.seed(seed)

    # Calculate the number of steps needed to reach end_size
    # Each iteration must progressively add less density in order for the
    # algorithm to remain efficient
    steps = 0
    current_size = initial_size
    while current_size < end_size:
        current_size *= upscale_factor
        steps += 1

    print(f"Steps: {steps}")

    # Compute the falloff curve given the number of steps
    curve = falloff_curve(steps, density_falloff_extremity, density_falloff_bias * steps)  # TODO: Add params

    # First we must set the central pixel to have maximum weight
    central_pixel = image.values[image.size // 2][image.size // 2]
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
            # The number of concurrent walkers is calculated based on density
            # estimation
            num_concurrent_walkers: int
            if use_concurrent_walkers:
                total_pixels = image.size**2
                frozen_pixels = image.density * total_pixels
                num_concurrent_walkers = max(1, int((step_density_threshold * total_pixels) - frozen_pixels))
            else:
                num_concurrent_walkers = 1

            simulate_random_walk(image, num_concurrent_walkers)

            # Update image density
            image.density = calculate_density(image)

        # Before we upscale the image, lets add the current copy to images to
        # track the upscale history
        images.append(copy.copy(image))

        # Upscale the image once step_density_threshold is met
        image = crisp_upscale(image, image.size * 2, upscale_jitter)

    return images


def traversable(image: Image) -> bool:
    def is_in_bounds(x, y):
        return 0 <= x < image.size and 0 <= y < image.size

    def bfs(start_pixel: Pixel, target_pixel: Pixel):
        queue = [start_pixel]
        visited = set()
        visited.add((start_pixel.x, start_pixel.y))

        while queue:
            current_pixel = queue.pop(0)
            if current_pixel == target_pixel:
                return True
            for direction in directions:
                nx, ny = (
                    current_pixel.x + direction[0],
                    current_pixel.y + direction[1],
                )
                if is_in_bounds(nx, ny):
                    neighbor = image.values[nx][ny]
                    if neighbor.frozen and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(neighbor)
        return False

    directions = (
        (0, -1),  # North
        (1, -1),  # North East
        (1, 0),  # East
        (1, 1),  # South East
        (0, 1),  # South
        (-1, 1),  # South West
        (-1, 0),  # West
        (-1, -1),  # North West
    )

    if not image.origin or not image.origin.frozen:
        return False

    # Find all frozen pixels
    frozen_pixels = [(x, y) for x in range(image.size) for y in range(image.size) if image.values[x][y].frozen]

    if not frozen_pixels:
        return False

    # Check if all frozen pixels can reach the origin
    origin_pixel = image.origin
    for x, y in frozen_pixels:
        if not bfs(image.values[x][y], origin_pixel):
            return False

    return True


def display_image(image: Image) -> None:
    weights = []
    for x in range(image.size):
        weights.append([])
        for y in range(image.size):
            weights[x].append(image.values[y][x].weight)
    weights = np.array(weights)

    plt.imshow(weights, cmap="terrain", origin="lower")
    plt.colorbar(label="Weight")
    plt.title("Heightmap Based on 2D Array of Weights")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.show()


def main():
    start_time = time.time()

    images = perform_dla(
        seed=2,
        initial_size=50,
        end_size=1000,
        initial_density_threshold=0.1,
        density_falloff_extremity=0.5,
        density_falloff_bias=1 / 3,
        use_concurrent_walkers=False,
        upscale_factor=2,
        upscale_jitter=1,
    )

    end_time = time.time()
    print(f"Image generation time: {end_time - start_time} seconds")

    if DEBUG:
        # Display image history
        for image in images:
            display_image(image)
            print(f"Image density: {image.density}")

        # Ensure that there is only one pixel with no outbound connections
        final_image = images[-1]
        print(
            final_image.origin.x,
            final_image.origin.y,
            final_image.origin.weight,
            final_image.origin.stuck_with,
        )
        inbound, outbound = get_connections(final_image)
        i = outbound.index([])
        print(f"({i // final_image.size}, {i % final_image.size}), {outbound.count([])}")
        print(traversable(final_image))


def set_pixel(image, x, y, sx, sy):
    pixel = image.values[x][y]
    pixel.frozen = True
    pixel.weight = 255
    pixel.stuck_with = image.values[sx][sy]


def test():
    image = Image(8)
    set_pixel(image, 4, 4, 4, 4)
    image.origin = image.values[4][4]
    image.values[4][4].stuck_with = None
    set_pixel(image, 5, 4, 4, 4)
    set_pixel(image, 5, 5, 5, 4)
    set_pixel(image, 3, 4, 4, 4)
    set_pixel(image, 3, 5, 3, 4)
    set_pixel(image, 2, 4, 3, 4)
    set_pixel(image, 6, 4, 5, 4)

    display_image(image)

    # random.seed(0)

    u1 = crisp_upscale(image, 8 * 3, 1)
    u2 = crisp_upscale(u1, 8 * 6, 1)
    u3 = crisp_upscale(u2, 8 * 9, 1)

    line_segments = find_straight_line_segments(u3)  # TODO: FIX THIS METHOD, ACCOUNT FOR DIAGONAL SEGMENTS
    print(line_segments)
    for line_segment in line_segments:
        (x0, y0), (x1, y1) = line_segment
        mx = (x0 + x1) // 2
        my = (y0 + y1) // 2
        u3.values[mx][my].weight = 200

        jx = random.randint(-2, 2)
        jy = random.randint(-2, 2)
        jmx = 0
        jmy = 0
        invalid = False

        if x0 == x1:
            # Vertical Line
            jmx, _ = constrain(mx + jx, 0, u3.size - 1)
            jmy, _ = constrain(my + jy, min(y0, y1), max(y0, y1))
        elif y0 == y1:
            # Horizontal
            jmx, _ = constrain(mx + jx, min(x0, x1), max(x0, x1))
            jmy, _ = constrain(my + jy, 0, u3.size - 1)

        u3.values[jmx][jmy].weight = 400

        b1 = draw_bresenham(x0, y0, jmx, jmy)
        b2 = draw_bresenham(jmx, jmy, x1, y1)
        bs = [b1, b2]

        # Before we check the validity of bresenham lines, unfreeze the pixels
        # in the line segment
        line_segment_points = draw_bresenham(x0, y0, x1, y1)
        for point in line_segment_points:
            u3.values[point[0]][point[1]].frozen = False

        for b in bs:
            for point in b:
                if u3.values[point[0]][point[1]].frozen:
                    invalid = True
                    break

        if invalid:
            # Re-freeze the points we unfroze
            for point in line_segment_points:
                u3.values[point[0]][point[1]].frozen = True
        else:
            for b in bs:
                for point in b[:-1]:
                    pixel = u3.values[point[0]][point[1]]
                    pixel.frozen = True
                    pixel.weight = 600
                    # TODO: FIX STUCK WITH FIELD

    i0, o0 = get_connections(image)
    i1, o1 = get_connections(u1)
    i2, o2 = get_connections(u2)
    i3, o3 = get_connections(u3)

    print([] in o0, [] in o1, [] in o2, [] in o3)

    index = o3.index([])
    print(f"({index // u3.size}, {index % u3.size})")
    print(len(line_segments))

    display_image(u1)
    display_image(u2)
    display_image(u3)


if __name__ == "__main__":
    test()
