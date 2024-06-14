from typing import List, Self, Tuple
import random
import math
import time

import numpy as np
import matplotlib.pyplot as plt

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
               14. We then use linear interpolation to upscale the Image
                   to the new size.
               15. Lastly, we use convolutions to assign each pixel a
                   weighted average of the adjacent pixels.
"""


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

    def __init__(self, size: int):
        self.size = size
        self.density = 0

        self.values = []
        for x in range(size):
            self.values.append([])
            for y in range(size):
                self.values[x].append(Pixel(x, y))


def constrain(
    value: int | float, low: int | float, high: int | float
) -> Tuple[bool, int | float]:
    temp = value
    value = min(value, high)
    value = max(value, low)
    return value == temp, value


def calculate_walk_probabilities(
    x: int, y: int, x_c: int, y_c: int
) -> Tuple[float, float, float, float]:
    # Calculate squared distances
    d_up = math.sqrt((x - x_c) ** 2 + (y - 1 - y_c) ** 2)
    d_down = math.sqrt((x - x_c) ** 2 + (y + 1 - y_c) ** 2)
    d_left = math.sqrt((x - 1 - x_c) ** 2 + (y - y_c) ** 2)
    d_right = math.sqrt((x + 1 - x_c) ** 2 + (y - y_c) ** 2)

    # Compute inverse squared distances
    w_up = 1 / d_up
    w_down = 1 / d_down
    w_left = 1 / d_left
    w_right = 1 / d_right

    # Sum of weights
    S = w_up + w_down + w_left + w_right

    # Normalize probabilities
    P_up = w_up / S
    P_down = w_down / S
    P_left = w_left / S
    P_right = w_right / S

    return P_up, P_down, P_left, P_right


def simulate_random_walk(
    image: Image, concurrent_walkers: int
) -> List[tuple[int, int]]:
    # TODO: Implement concurrent random walks
    # Place a pixel at a random position along an edge of the image
    edges = (
        ((0, image.size - 1), (0, 0)),  # Top
        ((0, image.size - 1), (image.size - 1, image.size - 1)),  # Bottom
        ((image.size - 1, image.size - 1), (0, image.size - 1)),  # Right
        ((0, 0), (0, image.size - 1)),  # Left
    )

    # Pick a random edge
    edge = random.choice(edges)

    # Define the directions and the probabilities of walking in a certain
    # direction dependent on the chosen edge
    directions = (
        (0, -1),  # North
        (0, 1),  # South
        (1, 0),  # East
        (-1, 0),  # West
    )

    x = random.randint(edge[0][0], edge[0][1])
    y = random.randint(edge[1][0], edge[1][1])

    # Is it really worth using walk_probabilities?
    walk_probabilities = calculate_walk_probabilities(
        x, y, image.size // 2, image.size // 2
    )

    # Perform random walk
    path = [(x, y)]
    while True:
        # Randomly choose a direction to walk in
        direction = random.choices(directions, weights=walk_probabilities, k=1)[0]
        # direction = random.choice(directions)
        _, x = constrain(x + direction[0], 0, image.size - 1)
        _, y = constrain(y + direction[1], 0, image.size - 1)

        # Once we find the coordinates of a frozen pixel, we will freeze the
        # previous pixel along the path of the random walk
        if image.values[x][y].frozen:
            prev_x = path[-1][0]
            prev_y = path[-1][1]

            pixel = image.values[prev_x][prev_y]
            pixel.stuck_with = image.values[x][y]
            pixel.frozen = True
            pixel.weight = 100
            break

        path.append((x, y))

    return path


def calculate_image_density(image: Image) -> float:
    frozen_pixels = sum(pixel.frozen for row in image.values for pixel in row)
    total_pixels = image.size**2
    return frozen_pixels / total_pixels


def crisp_upscale(image: Image, new_image_size: int, midpoint_jitter: int) -> Image:
    scale_factor = new_image_size / image.size
    new_image = Image(new_image_size)

    def draw_bresenham_line(x0, y0, x1, y1, weight):
        """
        Draw a line between (x0, y0) and (x1, y1) with the given weight and
        jittered midpoints.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        points = []

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        # Jitter midpoints and draw lines between them
        jittered_points = [points[0]]
        for i in range(1, len(points) - 1):
            x, y = points[i]
            jitter_x = x + random.randint(-midpoint_jitter, midpoint_jitter)
            jitter_y = y + random.randint(-midpoint_jitter, midpoint_jitter)
            _, jitter_x = constrain(jitter_x, 0, new_image.size - 1)
            _, jitter_y = constrain(jitter_y, 0, new_image.size - 1)
            jittered_points.append((jitter_x, jitter_y))
        jittered_points.append(points[-1])

        # Draw the lines with jittered points
        for j in range(len(jittered_points) - 1):
            x0, y0 = jittered_points[j]
            x1, y1 = jittered_points[j + 1]
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                new_image.values[x0][y0].weight = weight
                new_image.values[x0][y0].frozen = True
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

    for x in range(image.size):
        for y in range(image.size):
            pixel = image.values[x][y]
            if pixel.frozen:
                new_x = int(x * scale_factor)
                new_y = int(y * scale_factor)
                new_image.values[new_x][new_y].weight = pixel.weight
                new_image.values[new_x][new_y].frozen = True
                new_image.values[new_x][new_y].stuck_with = pixel.stuck_with

                # Draw connections
                if pixel.stuck_with:
                    old_stuck_x = pixel.stuck_with.x
                    old_stuck_y = pixel.stuck_with.y
                    new_stuck_x = int(old_stuck_x * scale_factor)
                    new_stuck_y = int(old_stuck_y * scale_factor)

                    # Draw a line between the points with jittered midpoints
                    draw_bresenham_line(
                        new_x, new_y, new_stuck_x, new_stuck_y, pixel.weight
                    )

    return new_image


def blurry_upscale(image: Image) -> Image:
    pass


def perform_dla(
    seed: int,
    initial_size: int,
    end_size: int,
    density_threshold: float,
    concurrent_walkers: int,
    upscale_jitter: int,
) -> Image:
    image = Image(initial_size)

    # Seed the random generator
    random.seed(seed)

    # First we must set the central pixel to have maximum weight
    central_pixel = image.values[image.size // 2][image.size // 2]
    central_pixel.weight = 255
    central_pixel.frozen = True
    image.density = calculate_image_density(image)

    # We need to keep simulating random walks until the image density has
    # passed the threshold
    while image.density < density_threshold:
        simulate_random_walk(image, concurrent_walkers)
        image.density = calculate_image_density(image)

    upscaled = crisp_upscale(image, end_size, upscale_jitter)
    return image, upscaled


def display_image(image: Image) -> None:
    weights = []
    for x in range(image.size):
        weights.append([])
        for y in range(image.size):
            weights[x].append(image.values[x][y].weight)
    weights = np.array(weights)

    plt.imshow(weights, cmap="terrain", origin="lower")
    plt.colorbar(label="Weight")
    plt.title("Heightmap Based on 2D Array of Weights")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.show()


def main():
    start_time = time.time()

    image, upscaled = perform_dla(0, 100, 1000, 0.1, 50, 5)
    display_image(image)
    display_image(upscaled)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
