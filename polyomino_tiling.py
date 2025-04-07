from itertools import product

def rotate(polyomino):
    """Generate all 90-degree rotations of a polyomino."""
    return [(-y, x) for x, y in polyomino]

def reflect(polyomino):
    """Generate the reflection of a polyomino."""
    return [(-x, y) for x, y in polyomino]

def generate_variants(polyomino):
    """Generate all unique rotations and reflections of a polyomino."""
    variants = set()
    current = polyomino
    for _ in range(4):
        current = rotate(current)
        variants.add(tuple(sorted(current)))
        variants.add(tuple(sorted(reflect(current))))
    return list(map(list, variants))

def can_place(grid, polyomino, x, y):
    """Check if a polyomino can be placed at (x, y) in the grid."""
    for dx, dy in polyomino:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < len(grid) and 0 <= ny < len(grid[0])) or grid[nx][ny]:
            return False
    return True

def place(grid, polyomino, x, y, value=True):
    """Place or remove a polyomino from the grid."""
    for dx, dy in polyomino:
        grid[x + dx][y + dy] = value

def get_cell_order(grid):
    """Sort cells by most-constrained heuristic: least available placements first."""
    order = []
    for x, y in product(range(len(grid)), range(len(grid[0]))):
        if not grid[x][y]:
            order.append((x, y))
    return order

def solve(grid, polyominoes, cell_order, index=0):
    """Recursive backtracking solver for polyomino tiling."""
    if index == len(cell_order):
        return True  # Successfully tiled
    x, y = cell_order[index]
    if grid[x][y]:
        return solve(grid, polyominoes, cell_order, index + 1)  # Skip filled cell
    for polyomino in polyominoes:
        if can_place(grid, polyomino, x, y):
            place(grid, polyomino, x, y, True)
            if solve(grid, polyominoes, cell_order, index + 1):
                return True
            place(grid, polyomino, x, y, False)  # Backtrack
    return False  # No valid placement found

def polyomino_tiling(grid_size, polyominoes):
    """Check if a grid can be tiled with given polyominoes."""
    grid = [[False] * grid_size[1] for _ in range(grid_size[0])]
    cell_order = get_cell_order(grid)
    return solve(grid, polyominoes, cell_order)

# Define base polyominoes
base_polyominoes = [
    [(0, 0), (1, 0), (1, 1)],  # L-tromino
    #[(0, 0), (1, 0), (1, -1)],  # Mirrored L-tromino
    [(0, 0), (0, 1), (1, 0)],  # T-tromino
    [(0, 0), (1, 0), (1, 1)]   # Z-tromino
]

# Generate all unique transformations
polyominoes = []
for poly in base_polyominoes:
    polyominoes.extend(generate_variants(poly))
polyominoes = list(set(tuple(sorted(p)) for p in polyominoes))

# Run search for increasing grid sizes
n = 1
while True:
    if polyomino_tiling((n, n), polyominoes):
        print(n)
    n += 1

