import copy
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

    def __init__(self, size: int):
        self.size = size
        self.density = 0

        self.values = []
        for x in range(size):
            self.values.append([])
            for y in range(size):
                self.values[x].append(Pixel(x, y))


def constrain(value: int | float, low: int | float, high: int | float) -> Tuple[int | float, bool]:
    return max(low, min(high, value)), value == max(low, min(high, value))


def calculate_density(image: Image) -> float:
    frozen_pixels = sum(pixel.frozen for row in image.values for pixel in row)
    total_pixels = image.size**2
    return frozen_pixels / total_pixels


def get_connections(image: Image) -> Tuple[List[List[int | None]], List[List[int | None]]]:
    inbound = [[] for _ in range(image.size**2)]
    outbound = [[] for _ in range(image.size**2)]

    for x in range(image.size):
        for y in range(image.size):
            index = x * image.size + y
            pixel = image.values[x][y]

            if pixel.frozen:
                if pixel.stuck_with:
                    stuck_index = pixel.stuck_with.x * image.size + pixel.stuck_with.y
                    inbound[stuck_index].append(index)
                    outbound[index].append(stuck_index)
                else:
                    outbound[index].append(None)  # No inbound connection
            else:  # Pixel is not frozen
                inbound[index].append(None)
                outbound[index].append(None)  # No outbound connection either

    return inbound, outbound


def simulate_random_walk(image: Image, num_concurrent_walkers: int):
    # Define edges and directions
    edges = (
        ((0, image.size - 1), (0, 0)),  # Top
        ((0, image.size - 1), (image.size - 1, image.size - 1)),  # Bottom
        ((image.size - 1, image.size - 1), (0, image.size - 1)),  # Right
        ((0, 0), (0, image.size - 1)),  # Left
    )

    directions = (
        (0, -1),  # North
        (0, 1),  # South
        (1, 0),  # East
        (-1, 0),  # West
    )

    # Create a list of walkers
    walkers = []
    for walker in range(num_concurrent_walkers):
        # Place a pixel at a random position along an edge of the image
        edge = random.choice(edges)
        x = random.randint(edge[0][0], edge[0][1])
        y = random.randint(edge[1][0], edge[1][1])

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
    # Create new image
    new_image = Image(new_image_size)

    def draw_bresenham(x0, y0, x1, y1):
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

    # Translate pixels from old image onto new image
    scale_factor = new_image_size / image.size
    for x in range(image.size):
        for y in range(image.size):
            if image.values[x][y].frozen:
                pixel = image.values[x][y]
                new_pixel = new_image.values[int(x * scale_factor)][int(y * scale_factor)]
                new_pixel.frozen = True
                new_pixel.weight = pixel.weight

    # We must now reconstruct every outbound connection from the original image
    inbound, outbound = get_connections(image)
    for i, outbound_connections in enumerate(outbound):
        if None not in outbound_connections:
            x0 = int((i // image.size) * scale_factor)
            y0 = int((i % image.size) * scale_factor)
            for outbound_connection in outbound_connections:
                x1 = int((outbound_connection // image.size) * scale_factor)
                y1 = int((outbound_connection % image.size) * scale_factor)

                if len(draw_bresenham(x0, y0, x1, y1)) > 2:
                    # Perform midpoint jittering
                    mx = (x0 + x1) // 2
                    my = (y0 + y1) // 2

                    # Choose one axis to jitter (to avoid line breakage)
                    jitter_axis = random.choice(["X", "Y"])
                    jitter = random.randint(-midpoint_jitter, midpoint_jitter)

                    if jitter_axis == "X":
                        _, mx_preserved = constrain(mx + jitter, 0, new_image.size)
                        if not mx_preserved:  # Jitter out of bounds, try the other axis
                            _, my_preserved = constrain(my + jitter, 0, new_image.size)
                            if my_preserved:
                                my += jitter
                            else:  # Both out of bounds, no jitter this time
                                pass
                        else:
                            mx += jitter
                    else:  # jitter_axis == "Y"
                        _, my_preserved = constrain(my + jitter, 0, new_image.size)
                        if not my_preserved:
                            _, mx_preserved = constrain(mx + jitter, 0, new_image.size)
                            if mx_preserved:
                                mx += jitter
                            else:
                                pass
                        else:
                            my += jitter

                    bresenham_lines = (draw_bresenham(x0, y0, mx, my), draw_bresenham(mx, my, x1, y1))
                else:
                    bresenham_lines = (draw_bresenham(x0, y0, x1, y1),)

                # Draw the bresenham lines
                for bresenham_line in bresenham_lines:
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


def soft_descent(x: int | float, a: int | float, k: int | float) -> float:
    return (math.e ** (-k * (x - (a / 2)))) / (1 + math.e ** (-k * (x - (a / 2))))


def perform_dla(
    seed: int,
    initial_size: int,
    end_size: int,
    initial_density_threshold: float,
    density_falloff_rate: float,
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

    # First we must set the central pixel to have maximum weight
    central_pixel = image.values[image.size // 2][image.size // 2]
    central_pixel.weight = 255
    central_pixel.frozen = True
    image.density = calculate_density(image)

    for step in range(steps):
        # Calculate the density threshold for this step
        step_density_threshold = soft_descent(step, steps, density_falloff_rate) * initial_density_threshold

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
        density_falloff_rate=2,
        use_concurrent_walkers=False,
        upscale_factor=2,
        upscale_jitter=1,
    )

    if DEBUG:
        # Display image history
        for image in images:
            display_image(image)
            print(f"Image density: {image.density}")

        # Ensure that there is only one pixel with no outbound connections
        final_image = images[-1]
        inbound, outbound = get_connections(final_image)
        for i, outbound_connection in enumerate(outbound):
            if len(outbound_connection) == 0:
                x = i // final_image.size
                y = i % final_image.size
                print(f"({x}, {y})")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")


def set_pixel(image, x, y, sx, sy):
    pixel = image.values[x][y]
    pixel.frozen = True
    pixel.weight = 255
    pixel.stuck_with = image.values[sx][sy]


def test():
    image = Image(8)
    set_pixel(image, 4, 4, 4, 4)
    image.values[4][4].stuck_with = None
    set_pixel(image, 5, 4, 4, 4)
    set_pixel(image, 5, 5, 5, 4)
    set_pixel(image, 3, 4, 4, 4)

    display_image(image)

    random.seed(0)

    u1 = crisp_upscale(image, 8 * 3, 1)
    u2 = crisp_upscale(u1, 8 * 6, 1)
    u3 = crisp_upscale(u2, 8 * 9, 1)

    display_image(u1)
    display_image(u2)
    display_image(u3)


if __name__ == "__main__":
    main()
