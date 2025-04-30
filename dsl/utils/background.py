import numpy as np


def find_background_colour(_grid: np.ndarray) -> int:
            """Return the most common colour in the grid."""
            unique, counts = np.unique(_grid, return_counts=True)
            return unique[np.argmax(counts)]

