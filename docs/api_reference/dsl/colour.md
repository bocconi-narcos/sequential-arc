# Colour Selection

Colour selection is the first step in the action pipeline. The agent chooses a colour from the grid, which is then used by the selection and transformation functions.

## Available Colour Selectors

The following colour selectors are available:

* **`colour(colour)`:** Selects a specific colour.
* **`most_common()`:** Selects the colour that occurs most frequently in the grid.
* **`least_common()`:** Selects the rarest colour that is present in the grid.
* **`nth_most_common(rank)`:** Selects the nth most common colour in the grid.
* **`nth_most_independent(rank, connectivity)`:** Selects the colour with the nth most independent cells.
* **`colour_of_nth_largest_shape(rank)`:** Selects the colour of the nth largest shape in the grid.
