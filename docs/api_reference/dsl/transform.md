# Transformation

Transformation is the final step in the action pipeline. The agent applies a transformation to the selected cells.

## Available Transformers

The following transformers are available:

* **`identity()`:** Does nothing.
* **`clear()`:** Clears the selected cells.
* **`new_colour(color)`:** Changes the colour of the selected cells to a new colour.
* **`background_colour()`:** Changes the colour of the selected cells to the background colour.
* **`invert_colors()`:** Inverts the colours of the selected cells.
* **`flip(axis)`:** Flips the selected cells horizontally or vertically.
* **`rotate(k)`:** Rotates the selected cells by 90, 180, or 270 degrees.
* **`slide_new(direction, mode, continuous, obstacles, fluid, superfluid)`:** Slides the selected cells in a given direction.
* **`slide_old(direction, mode, continuous, obstacles, fluid, superfluid)`:** Slides the selected cells in a given direction (old implementation).
