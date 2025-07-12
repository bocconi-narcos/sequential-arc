# Selection

Selection is the second step in the action pipeline. The agent selects a set of cells from the grid, which is then used by the transformation function.

## Available Selectors

The following selectors are available:

* **`all_cells()`:** Selects all cells in the grid.
* **`colour(colour)`:** Selects all cells of a specific colour.
* **`components4(colour)`:** Selects all 4-connected components of a specific colour.
* **`components8(colour)`:** Selects all 8-connected components of a specific colour.
* **`nth_largest_shape(colour, connectivity, rank)`:** Selects the nth largest shape of a specific colour.
* **`independent_cells(colour, connectivity)`:** Selects all independent cells of a specific colour.
* **`outer_border4(colour)`:** Selects the outer border of all 4-connected components of a specific colour.
* **`inner_border4(colour)`:** Selects the inner border of all 4-connected components of a specific colour.
* **`grid_border(colour)`:** Selects the border of the grid.
* **`outer_border8(colour)`:** Selects the outer border of all 8-connected components of a specific colour.
* **`inner_border8(colour)`:** Selects the inner border of all 8-connected components of a specific colour.
* **`adjacent4(colour, contacts)`:** Selects all cells that are 4-adjacent to a specific colour.
* **`adjacent8(colour, contacts)`:** Selects all cells that are 8-adjacent to a specific colour.
* **`rectangles(colour, height, width)`:** Selects all rectangles of a specific size and colour.
