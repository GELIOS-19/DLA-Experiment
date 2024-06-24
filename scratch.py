import matplotlib.pyplot as plt


def bezier_sigmoid(a, m, b, precision=10000):
    def bezier_curve(t, p0, p1, p2, p3):
        return ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0],
                (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1])

    def vertical_line_test(points):
        x = 0
        for point in points:
            if point[0] > x:
                x = point[0]
            elif point[0] < x:
                return False
        return True

    # Define control points
    p0 = (0.0, 1.0)
    p3 = (a, 0.0)

    control_line = p0[1] - m * p0[0]
    initial_line_x = (1 - control_line) * m
    final_line_x = (0 - control_line) * m
    midpoint = (initial_line_x + final_line_x) / 2
    offset = b - midpoint
    initial_line_x += offset
    final_line_x += offset

    p1 = (initial_line_x, 1.0)
    p2 = (final_line_x, 0.0)

    # Find the bezier points
    bezier_points = [bezier_curve(t / precision, p0, p1, p2, p3) for t in range(precision + 1)]

    # Check if curve passes the vertical line test
    if not vertical_line_test(bezier_points):
        raise ArithmeticError(f"For the given parameters (a={a}, m={m}, b={b}), a function cannot be formed.")

    return bezier_points

a = 4
m = 2
b = 2
points = bezier_sigmoid(a, m, b)

x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]

plt.plot(x_coords, y_coords, label=f'a={a}, m={m}, b={b}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Bezier Sigmoid Curve')
plt.legend()
plt.grid(True)
plt.show()