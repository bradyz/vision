import numpy as np

from matplotlib import pyplot as plt


def get_left_right_up_down(i, j):
    left, right = (i + 0.5, j), (i + 0.5, j + 1.0)
    up, down = (i, j + 0.5), (i + 1.0, j + 0.5)

    return left, right, up, down


def do_case(case, i, j, segments):
    l, r, u, d = get_left_right_up_down(i, j)

    if case == '0000':
        pass
    elif case == '1000':
        segments.append((l, u))
    elif case == '0100':
        segments.append((u, r))
    elif case == '0010':
        segments.append((l, d))
    elif case == '0001':
        segments.append((d, r))
    elif case == '1100':
        segments.append((l, r))
    elif case == '0011':
        segments.append((l, r))
    elif case == '1010':
        segments.append((u, d))
    elif case == '0101':
        segments.append((u, d))
    elif case == '1001':
        segments.append((l, d))
        segments.append((u, r))
    elif case == '0110':
        segments.append((l, u))
        segments.append((d, r))
    elif case == '1110':
        segments.append((d, r))
    elif case == '1101':
        segments.append((l, d))
    elif case == '1011':
        segments.append((u, r))
    elif case == '0111':
        segments.append((l, u))
    elif case == '1111':
        pass


def march(grid, m, n):
    segments = list()

    for i in range(m-1):
        for j in range(n-1):
            # Turn the corners into a string.
            a, b = grid[i,j] > 0.0, grid[i,j+1] > 0.0
            c, d = grid[i+1,j] > 0.0, grid[i+1,j+1] > 0.0

            # A string of 1's and 0's.
            case = ''.join(map(lambda x: str(int(x)), [a, b, c, d]))

            # Death by 2^neighbors cases.
            do_case(case, i, j, segments)

    return segments


def main():
    m, n = map(int, input().split())

    grid = [list(map(int, input().split())) for _ in range(m)]
    grid = np.array(grid)
    grid = np.pad(grid, ((1, 1), (1, 1)), 'constant')

    # Rows and columns, add two for padding.
    m += 2
    n += 2

    segments = march(grid, m, n)

    # Show the original indicator function.

    for i in range(m):
        for j in range(n):
            if grid[i,j] > 0.0:
                plt.plot(i, j, 'ro')

    # Show the surface.

    for p1, p2 in segments:
        x1, y1 = p1
        x2, y2 = p2

        plt.plot([x1, x2], [y1, y2], 'r--')

    plt.show()


if __name__ == '__main__':
    main()
