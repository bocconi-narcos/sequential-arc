import unittest
import numpy as np
from dsl.colour import ColorSelector
from dsl.select import GridSelector # Needed for nth_most_independent

class TestColorSelectorSpecific(unittest.TestCase):

    def setUp(self):
        self.grid_empty = np.array([[]], dtype=int)
        self.grid_0x0 = np.empty((0,0), dtype=int)
        self.grid_1x1_c0 = np.array([[0]])
        self.grid_all_unique = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        self.grid_mostly_1 = np.array([[1, 1, 2], [1, 0, 3], [1, 1, 1]]) # 1:6, 0:1, 2:1, 3:1
        self.grid_tie_most = np.array([[1, 1, 2, 2], [0, 0, 3, 4]]) # 1:2, 2:2, 0:2, 3:1, 4:1. most_common -> 0
        self.grid_tie_least = np.array([[1, 1, 1, 2, 2, 2, 3, 4]]) # 3:1, 4:1. least_common -> 3
        self.grid_all_5 = np.array([[5,5],[5,5]])

    def assertColorInRange(self, color, min_val=0, max_val=9):
        self.assertIsInstance(color, (int, np.integer))
        self.assertTrue(min_val <= color <= max_val, f"Color {color} out of range {min_val}-{max_val}")

    # --- Test ColorSelector.colour and partials ---
    def test_colour_and_partials(self):
        grid = self.grid_mostly_1
        for i in range(11): # Test colour_0 to colour_10
            with self.subTest(colour_val=i):
                res = ColorSelector.colour(grid, colour=i)
                self.assertEqual(res, i)
                self.assertColorInRange(res, 0, 10) # Allows up to 10

                # Test partials if they exist
                if hasattr(ColorSelector, f"colour_{i}"):
                    partial_func = getattr(ColorSelector, f"colour_{i}")
                    self.assertEqual(partial_func(grid), i)

        with self.assertRaisesRegex(ValueError, "colour must be a non-negative integer"):
            ColorSelector.colour(grid, colour=-1)
        with self.assertRaisesRegex(ValueError, "colour must be a non-negative integer"):
            ColorSelector.colour(grid, colour=3.5)
        with self.assertRaisesRegex(ValueError, "colour must be a non-negative integer"):
            ColorSelector.colour(grid, colour="abc")


    # --- Test ColorSelector.most_common ---
    def test_most_common_specific(self):
        self.assertEqual(ColorSelector.most_common(self.grid_mostly_1), 1)
        self.assertEqual(ColorSelector.most_common(self.grid_tie_most), 0) # argmax picks smallest index in tie
        self.assertEqual(ColorSelector.most_common(self.grid_all_unique), 0) # argmax picks smallest index if all counts are 1
        self.assertEqual(ColorSelector.most_common(self.grid_1x1_c0), 0)
        self.assertEqual(ColorSelector.most_common(self.grid_all_5), 5)
        # Test with a grid that has colors outside 0-9 range if num_colours allows
        grid_high_color = np.array([[12, 12, 1]])
        # most_common uses minlength=num_colours (10), so 12 is out of argmax range unless it's the only thing
        # If grid_high_color.ravel() is [12,12,1], bincount(minlength=10) is [0,1,0,0,0,0,0,0,0,0], argmax is 1.
        # If bincount was on grid.ravel() directly without minlength, it could be different.
        # Current implementation:
        self.assertEqual(ColorSelector.most_common(grid_high_color), 1)
        grid_only_high = np.array([[12,12,12]]) # bincount(minlength=10) -> [0,0,...] -> argmax is 0
        self.assertEqual(ColorSelector.most_common(grid_only_high), 0)

        # Empty grid - np.bincount([]) is [], .argmax() is error.
        # However, grid.ravel() for (0,0) is [], for (1,0) is [].
        # np.bincount([], minlength=10) is [0,0,0,0,0,0,0,0,0,0], argmax is 0.
        self.assertEqual(ColorSelector.most_common(self.grid_0x0), 0)


    # --- Test ColorSelector.least_common ---
    def test_least_common_specific(self):
        self.assertEqual(ColorSelector.least_common(self.grid_mostly_1), 0) # 0,2,3 all appear once, 0 is smallest index
        self.assertEqual(ColorSelector.least_common(self.grid_tie_least), 3) # 3 and 4 appear once, 3 is smaller

        # Grid with some colors absent
        grid_absent = np.array([[2,2,3],[3,4,4]]) # 0,1,5-9 absent. Present: 2 (2), 3 (2), 4 (2). Smallest is 2.
        self.assertEqual(ColorSelector.least_common(grid_absent), 2)

        grid_all_equal_present = np.array([[0,1,2],[3,4,5],[6,7,8]]) # Color 9 is missing
        # counts are [1,1,1,1,1,1,1,1,1,0]. counts[counts==0]=BIG. Smallest is 0.
        self.assertEqual(ColorSelector.least_common(grid_all_equal_present), 0)

        self.assertEqual(ColorSelector.least_common(self.grid_1x1_c0), 0)
        self.assertEqual(ColorSelector.least_common(self.grid_all_5), 5)

        # Grid where all standard colors 0-9 are absent
        grid_only_high = np.array([[10, 11, 10]]) # Assuming num_colours = 10
        # counts = [0,0,0,0,0,0,0,0,0,0] (for 0-9). All become BIG. argmin is 0.
        self.assertEqual(ColorSelector.least_common(grid_only_high), 0)

        # Empty grid (0,0) -> bincount is all zeros, then all BIG, argmin is 0
        self.assertEqual(ColorSelector.least_common(self.grid_0x0), 0)


    # --- Test ColorSelector.nth_most_common ---
    def test_nth_most_common_specific(self):
        # Freq: 1 (6), 0 (1), 2 (1), 3 (1)
        # Order: 1, 0, 2, 3 (due to tie-breaking by index for 0,2,3)
        grid = self.grid_mostly_1
        self.assertEqual(ColorSelector.nth_most_common(grid, rank=0), 1)
        self.assertEqual(ColorSelector.nth_most_common(grid, rank=1), 0)
        self.assertEqual(ColorSelector.nth_most_common(grid, rank=2), 2)
        self.assertEqual(ColorSelector.nth_most_common(grid, rank=3), 3)

        # Test partials
        self.assertEqual(ColorSelector.second_most_common(grid), 0)
        self.assertEqual(ColorSelector.third_most_common(grid), 2)

        # Rank greater than number of unique colors
        # Unique colors: 0,1,2,3 (4 unique). rank=4 should return least_common.
        # least_common for this grid is 0.
        self.assertEqual(ColorSelector.nth_most_common(grid, rank=4), ColorSelector.least_common(grid))
        self.assertEqual(ColorSelector.nth_most_common(grid, rank=10), ColorSelector.least_common(grid))

        with self.assertRaisesRegex(ValueError, "rank must be non-negative"):
            ColorSelector.nth_most_common(grid, rank=-1)

        # Tie breaking in nth_most_common: argsort is stable for equal elements,
        # so original order of indices for ties is preserved.
        # grid_tie_most = np.array([[1, 1, 2, 2], [0, 0, 3, 4]])
        # counts: 0:2, 1:2, 2:2, 3:1, 4:1 (minlength 10)
        # sorted counts desc: [2,2,2,1,1,0,0,0,0,0]
        # order (indices of sorted counts): [0,1,2,3,4,5,6,7,8,9] (default if all counts unique)
        # argsort(counts) -> [5,6,7,8,9,3,4,0,1,2] (indices that would sort counts asc)
        # argsort(counts)[::-1] -> [2,1,0,4,3,9,8,7,6,5] (indices that sort counts desc)
        # So: rank 0 -> color 2, rank 1 -> color 1, rank 2 -> color 0
        self.assertEqual(ColorSelector.nth_most_common(self.grid_tie_most, rank=0), 0) # Should be 0, 1, or 2. np.argsort sorts by value, then by index for ties.
                                                                                      # Counts: c[0]=2, c[1]=2, c[2]=2, c[3]=1, c[4]=1
                                                                                      # argsort(counts) gives indices that sort counts: [ (idx for count 0), ..., (idx for count 2)]
                                                                                      # For this grid, counts are [2,2,2,1,1,0,0,0,0,0]
                                                                                      # argsort(counts) -> [5,6,7,8,9,3,4,0,1,2] (original indices of sorted items)
                                                                                      # [::-1] -> [2,1,0,4,3,9,8,7,6,5]
        self.assertEqual(ColorSelector.nth_most_common(self.grid_tie_most, rank=0), 2) # Color 2
        self.assertEqual(ColorSelector.nth_most_common(self.grid_tie_most, rank=1), 1) # Color 1
        self.assertEqual(ColorSelector.nth_most_common(self.grid_tie_most, rank=2), 0) # Color 0
        self.assertEqual(ColorSelector.nth_most_common(self.grid_tie_most, rank=3), 4) # Color 4
        self.assertEqual(ColorSelector.nth_most_common(self.grid_tie_most, rank=4), 3) # Color 3


    # --- Test ColorSelector.nth_most_independent ---
    def test_nth_most_independent_specific(self):
        # Grid setup for independent cells:
        # Background is 0 (most common)
        # 1s: two independent (1x1) cells
        # 2s: one independent (1x1) cell
        # 3s: zero independent cells (forms a 2x1 block)
        # 4s: one independent cell, but it's background color by count (0)
        grid_indep = np.array([
            [1, 0, 2, 0, 1],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 0] # Assume 4 is independent if GridSelector.independent_cells checks against actual background
        ])
        # find_background_colour for grid_indep is 0.
        # GridSelector.independent_cells(grid_indep, 1, 4) -> 2 cells (True at [0,0], [0,4]) -> count 2
        # GridSelector.independent_cells(grid_indep, 2, 4) -> 1 cell (True at [0,2]) -> count 1
        # GridSelector.independent_cells(grid_indep, 3, 4) -> 0 cells
        # GridSelector.independent_cells(grid_indep, 4, 4) -> 1 cell (True at [2,3]) -> count 1
        # GridSelector.independent_cells(grid_indep, 0, 4) -> 0 cells (cannot be independent and background)
        # Counts: c1:2, c2:1, c3:0, c4:1, c0:0. Others: 0
        # Order by count desc (tie break by color index asc): 1 (2), 2 (1), 4 (1), 0 (0), 3 (0) ...
        # Resulting color order from argsort: [1, 2, 4, 0, 3, 5, 6, 7, 8, 9] (hypothetical, depends on GridSelector)

        # Based on code: order = np.argsort(counts)[::-1]
        # counts for colors 0-9: [0, 2, 1, 0, 1, 0, 0, 0, 0, 0]
        # argsort(counts) -> [0,2,3,5,6,7,8,9,  1,4] (indices that sort counts: 0,0,0,0,0,0,0,0, 1,1, 2) incorrect logic
        # argsort(counts) -> [0,3,5,6,7,8,9, 2,4, 1] for counts [0,2,1,0,1,0,0,0,0,0]
        # [::-1] -> [1, 4,2, 9,8,7,6,5,3,0]
        # rank 0 -> color 1
        # rank 1 -> color 4
        # rank 2 -> color 2
        # rank 3 -> color 9 (count 0)
        # rank -1 (last) -> color 0 (count 0)

        self.assertEqual(ColorSelector.nth_most_independent(grid_indep, rank=0), 1)
        self.assertEqual(ColorSelector.nth_most_independent(grid_indep, rank=1), 4)
        self.assertEqual(ColorSelector.nth_most_independent(grid_indep, rank=2), 2)
        # Partials
        self.assertEqual(ColorSelector.most_independent_cells(grid_indep), 1)
        self.assertEqual(ColorSelector.second_most_independent_cells(grid_indep), 4)

        # Rank out of bounds (idx = -1)
        self.assertEqual(ColorSelector.nth_most_independent(grid_indep, rank=10), 0) # Smallest index with min count

        with self.assertRaisesRegex(ValueError, "rank must be non-negative"):
            ColorSelector.nth_most_independent(grid_indep, rank=-1)

        grid_none_indep = np.array([[1,1],[1,1]]) # No independent cells if background is different (e.g. not present)
                                                  # If background is 1, then no non-background.
                                                  # Assuming background is not 1.
        # If all counts are 0, argsort is [0,1,2...], [::-1] is [9,8,...]. rank 0 is 9.
        self.assertEqual(ColorSelector.nth_most_independent(grid_none_indep, rank=0), 9) # Or 0 if all counts are 0 (depends on tie break)


    # --- Test ColorSelector.colour_of_nth_largest_shape ---
    def test_colour_of_nth_largest_shape_specific(self):
        grid = np.array([
            [1, 1, 0, 2, 2],  # Shape1 (1): size 2. Shape2 (2): size 2
            [1, 0, 0, 0, 2],
            [3, 3, 3, 0, 0],  # Shape3 (3): size 3
            [4, 4, 4, 4, 0],  # Shape4 (4): size 4
            [0, 0, 0, 5, 0]   # Shape5 (5): size 1
        ])
        # Sizes: 4 (col 4), 3 (col 3), 2 (col 1), 2 (col 2), 1 (col 5)
        # Sorted by size desc: (4,c4), (3,c3), (2,c1), (2,c2), (1,c5) (tie break by color not specified, depends on np.unique and loop order)
        # np.unique(grid) -> [0,1,2,3,4,5]. Loop order: 0,1,2,3,4,5
        # shapes: (c1,s2), (c2,s2), (c3,s3), (c4,s4), (c5,s1)
        # sizes: [2,2,3,4,1], colours_of_shapes: [1,2,3,4,5]
        # order (indices of sizes sorted desc): [3 (size 4), 2 (size 3), 0 (size 2), 1 (size 2), 4 (size 1)]
        # Corresp colours: [4, 3, 1, 2, 5]

        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(grid, rank=0), 4)
        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(grid, rank=1), 3)
        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(grid, rank=2), 1) # Tie break for size 2: color 1 comes before 2
        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(grid, rank=3), 2)
        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(grid, rank=4), 5)

        # Rank out of bounds (idx = order[-1]) -> should be smallest shape's color
        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(grid, rank=10), 5)

        with self.assertRaisesRegex(ValueError, "rank must be non-negative"):
            ColorSelector.colour_of_nth_largest_shape(grid, rank=-1)

        grid_no_shapes = np.array([[0,0],[0,0]]) # All one color (bg), no distinct shapes found by label on mask
        # sizes list will be empty. Returns most_common.
        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(grid_no_shapes, rank=0), 0)

        grid_empty_shapes = self.grid_0x0
        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(grid_empty_shapes, rank=0), 0) # most_common of empty is 0.

    # --- Input Validation Tests ---
    def test_input_validation_specific(self):
        selectors_to_test = [
            ColorSelector.most_common,
            ColorSelector.least_common,
            lambda g: ColorSelector.colour(g, colour=0),
            lambda g: ColorSelector.nth_most_common(g, rank=0),
            lambda g: ColorSelector.nth_most_independent(g, rank=0),
            lambda g: ColorSelector.colour_of_nth_largest_shape(g, rank=0),
        ]

        invalid_inputs = [
            ([1,2,3]), # Python list
            (np.array([1,2,3])), # 1D numpy array
            (np.array([[[1],[2]],[[3],[4]]])), # 3D numpy array
            (np.array([["a","b"],["c","d"]])), # Non-integer dtype
            None
        ]

        for selector_func in selectors_to_test:
            for i, invalid_grid in enumerate(invalid_inputs):
                with self.subTest(func_name=selector_func.__name__, input_idx=i):
                    with self.assertRaises(ValueError):
                        selector_func(invalid_grid)

        # Test _check_grid directly for sanity (though it's private)
        with self.assertRaisesRegex(ValueError, "grid must be a 2-D NumPy array"):
            ColorSelector._check_grid(np.array([1,2,3]))
        with self.assertRaisesRegex(ValueError, "grid dtype must be an integer type"):
            ColorSelector._check_grid(np.array([["a"]]))


if __name__ == '__main__':
    unittest.main()
