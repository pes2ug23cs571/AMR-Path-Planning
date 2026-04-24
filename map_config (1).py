import numpy as np

ROWS, COLS = 20, 20
CELL_SIZE  = 32
SENSOR_RANGE = 3

GRID = np.zeros((ROWS, COLS), dtype=np.int8)

# Outer border walls
GRID[0, :]  = 1
GRID[-1, :] = 1
GRID[:, 0]  = 1
GRID[:, -1] = 1

# Inner obstacles — room 1
GRID[2, 2:8] = 1
GRID[2:6, 2] = 1
GRID[2:6, 7] = 1

# Inner obstacles — room 2
GRID[4, 10:15] = 1
GRID[4:8, 10]  = 1
GRID[4:8, 14]  = 1

# Corridor wall
GRID[10, 3:17] = 1
GRID[10, 9]    = 0

# Extra pillars
GRID[14:17, 5]  = 1
GRID[14:17, 15] = 1