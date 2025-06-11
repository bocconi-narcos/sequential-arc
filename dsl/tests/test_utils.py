import unittest
import numpy as np
from dsl.utils.background import find_background_colour
from dsl.utils.padding import pad_grid, unpad_grid

class TestDSLUtils(unittest.TestCase):

    # --- Tests for find_background_colour ---
    def test_find_background_colour_specific(self):
        grid_clear_bg = np.array([[0, 0, 1], [0, 2, 0]], dtype=int)  # Background is 0
        self.assertEqual(find_background_colour(grid_clear_bg), 0)

        grid_predominant_bg = np.array([[1, 5, 5], [5, 1, 5], [2, 5, 5]], dtype=int) # Background is 5
        self.assertEqual(find_background_colour(grid_predominant_bg), 5)

        grid_all_unique = np.array([[1, 2], [3, 4]], dtype=int) # All unique, np.argmax picks first of max count -> smallest value (1)
        self.assertEqual(find_background_colour(grid_all_unique), 1)

        grid_tie_bg = np.array([[1,1,2,2,0]], dtype=int) # Counts: 1 (2), 2 (2), 0(1). Tie between 1 and 2. np.argmax picks first -> 1
        self.assertEqual(find_background_colour(grid_tie_bg), 1)


        grid_one_color = np.array([[7, 7], [7, 7]], dtype=int) # Background is 7
        self.assertEqual(find_background_colour(grid_one_color), 7)

        grid_1x1 = np.array([[3]], dtype=int)
        self.assertEqual(find_background_colour(grid_1x1), 3)

        # Test empty grids - current implementation raises ValueError via np.argmax
        grid_empty_0x0 = np.empty((0, 0), dtype=int)
        with self.assertRaises(ValueError, msg="find_background_colour on (0,0) grid should raise ValueError"):
            find_background_colour(grid_empty_0x0)

        grid_empty_0xN = np.empty((0, 3), dtype=int)
        with self.assertRaises(ValueError, msg="find_background_colour on (0,N) grid should raise ValueError"):
            find_background_colour(grid_empty_0xN)

        grid_empty_Nx0 = np.empty((3, 0), dtype=int)
        # np.unique on (N,0) grid.ravel() is empty.
        with self.assertRaises(ValueError, msg="find_background_colour on (N,0) grid should raise ValueError"):
            find_background_colour(grid_empty_Nx0)


    def test_find_background_colour_input_validation(self):
        # These might be caught by general DSL tests if those pass numpy arrays directly.
        # However, testing directly ensures this util is robust.
        # Based on current `find_background_colour`, it doesn't have its own _check_grid.
        # It relies on np.unique, which can handle some non-2D arrays.
        # The problem description implies it should work on 2D grids.
        # If strict 2D is required, _check_grid should be added to it.
        # For now, testing numpy's behavior.

        # Non-2D array (1D) - np.unique handles this
        grid_1d = np.array([1,1,0,0,0])
        self.assertEqual(find_background_colour(grid_1d), 0)

        # Non-integer dtype array - np.unique handles this and maintains dtype for unique
        # but argmax will still work on counts.
        grid_float = np.array([[1.0, 1.0, 0.0],[0.0,0.0,1.0]])
        self.assertEqual(find_background_colour(grid_float), 0.0) # Returns float if input is float

        # Test with non-NumPy array (list of lists) - np.unique handles this
        list_grid = [[1,1,0],[0,0,0]]
        self.assertEqual(find_background_colour(list_grid), 0)


    # --- Tests for pad_grid ---
    def test_pad_grid_specific(self):
        grid_small = np.array([[1, 2], [3, 4]], dtype=int)

        # Pad to larger shape, default fill_val (-1)
        padded_default = pad_grid(grid_small, target_shape=(3, 4))
        expected_default = np.array([[1, 2, -1, -1], [3, 4, -1, -1], [-1, -1, -1, -1]], dtype=int)
        np.testing.assert_array_equal(padded_default, expected_default)
        self.assertEqual(padded_default.shape, (3, 4))
        self.assertEqual(padded_default.dtype, grid_small.dtype)

        # Pad with custom fill_val
        padded_custom_fill = pad_grid(grid_small, target_shape=(3, 3), fill_val=0)
        expected_custom_fill = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]], dtype=int)
        np.testing.assert_array_equal(padded_custom_fill, expected_custom_fill)
        self.assertEqual(padded_custom_fill.dtype, grid_small.dtype)

        # Target shape same as grid shape
        padded_same_shape = pad_grid(grid_small, target_shape=(2, 2), fill_val=0)
        np.testing.assert_array_equal(padded_same_shape, grid_small, "Should be a copy on new canvas")
        self.assertIsNot(padded_same_shape, grid_small, "Should be a new object")
        self.assertEqual(padded_same_shape.dtype, grid_small.dtype)

        # Target shape with one dimension larger
        padded_one_dim = pad_grid(grid_small, target_shape=(2, 3), fill_val=5)
        expected_one_dim = np.array([[1, 2, 5], [3, 4, 5]], dtype=int)
        np.testing.assert_array_equal(padded_one_dim, expected_one_dim)
        self.assertEqual(padded_one_dim.dtype, grid_small.dtype)

        # ValueError if target_shape is smaller
        with self.assertRaisesRegex(ValueError, "Grid .* larger than canvas"):
            pad_grid(grid_small, target_shape=(1, 2))
        with self.assertRaisesRegex(ValueError, "Grid .* larger than canvas"):
            pad_grid(grid_small, target_shape=(2, 1))
        with self.assertRaisesRegex(ValueError, "Grid .* larger than canvas"):
            pad_grid(grid_small, target_shape=(1, 1))

        # Padding an empty grid (0,0)
        grid_empty_0x0 = np.empty((0, 0), dtype=np.int8) # use specific dtype
        padded_empty = pad_grid(grid_empty_0x0, target_shape=(2, 2), fill_val=7)
        expected_empty_padded = np.full((2, 2), 7, dtype=np.int8)
        np.testing.assert_array_equal(padded_empty, expected_empty_padded)
        self.assertEqual(padded_empty.dtype, grid_empty_0x0.dtype)

        # Padding a (0,N) grid
        grid_empty_0xN = np.empty((0,3), dtype=int)
        padded_0xN = pad_grid(grid_empty_0xN, target_shape=(2,3), fill_val=8)
        expected_0xN_padded = np.full((2,3), 8, dtype=int)
        np.testing.assert_array_equal(padded_0xN, expected_0xN_padded)


    # --- Tests for unpad_grid ---
    def test_unpad_grid_specific(self):
        # Grid with padding on all sides, default fill_val (-1)
        grid_padded_default = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=int)
        unpadded_default = unpad_grid(grid_padded_default)
        expected_unpadded_default = np.array([[1]], dtype=int)
        np.testing.assert_array_equal(unpadded_default, expected_unpadded_default)

        # Grid with padding on some sides, custom fill_val (0)
        grid_padded_custom = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0]], dtype=int)
        unpadded_custom = unpad_grid(grid_padded_custom, fill_val=0)
        expected_unpadded_custom = np.array([[1, 2], [3, 4]], dtype=int)
        np.testing.assert_array_equal(unpadded_custom, expected_unpadded_custom)

        # Grid with no padding
        grid_no_padding = np.array([[1, 0], [0, 1]], dtype=int)
        unpadded_no_padding = unpad_grid(grid_no_padding, fill_val=-5) # Fill val not present
        np.testing.assert_array_equal(unpadded_no_padding, grid_no_padding)
        # Check if it might return a view (hard to assert reliably, but can check object identity)
        # self.assertIsNot(unpadded_no_padding, grid_no_padding) # This might fail if it's a view

        # Grid entirely made of fill_val
        grid_all_fill = np.full((3, 3), 5, dtype=int)
        unpadded_all_fill = unpad_grid(grid_all_fill, fill_val=5)
        self.assertEqual(unpadded_all_fill.shape, (0, 0))

        # Grid that becomes empty after unpadding (e.g. single row/col of actual content)
        grid_becomes_empty_row = np.array([[-1, -1, -1], [1, 2, 3], [-1, -1, -1]], dtype=int)
        unpadded_becomes_empty_row = unpad_grid(grid_becomes_empty_row, fill_val=-1)
        self.assertEqual(unpadded_becomes_empty_row.shape, (1,3)) # Content is [[1,2,3]]
        np.testing.assert_array_equal(unpadded_becomes_empty_row, np.array([[1,2,3]]))


        grid_becomes_empty_col = np.array([[-1, 1, -1], [-1, 2, -1], [-1, 3, -1]], dtype=int)
        unpadded_becomes_empty_col = unpad_grid(grid_becomes_empty_col, fill_val=-1)
        self.assertEqual(unpadded_becomes_empty_col.shape, (3,1)) # Content is [[1],[2],[3]]
        np.testing.assert_array_equal(unpadded_becomes_empty_col, np.array([[1],[2],[3]]))

        # Grid with internal areas of fill_val (should not be removed)
        grid_internal_fill = np.array([[1, 2, 3], [4, -1, 5], [-1, -1, -1]], dtype=int) # Last row is padding
        unpadded_internal = unpad_grid(grid_internal_fill, fill_val=-1)
        expected_internal_unpadded = np.array([[1, 2, 3], [4, -1, 5]], dtype=int)
        np.testing.assert_array_equal(unpadded_internal, expected_internal_unpadded)

        # Unpadding an already empty grid (0,0)
        grid_empty_0x0 = np.empty((0, 0), dtype=int)
        unpadded_empty_0x0 = unpad_grid(grid_empty_0x0, fill_val=-1)
        self.assertEqual(unpadded_empty_0x0.shape, (0, 0))

        # Unpadding (0,N) or (N,0)
        grid_empty_0xN = np.empty((0,3), dtype=int)
        unpadded_empty_0xN = unpad_grid(grid_empty_0xN, fill_val=-1)
        self.assertEqual(unpadded_empty_0xN.shape, (0,0)) # Becomes (0,0) as no rows to keep

        grid_empty_Nx0 = np.empty((3,0), dtype=int)
        unpadded_empty_Nx0 = unpad_grid(grid_empty_Nx0, fill_val=-1)
        self.assertEqual(unpadded_empty_Nx0.shape, (0,0)) # Becomes (0,0) as no cols to keep

        # Grid that is all fill value except one cell
        grid_one_cell_not_fill = np.full((3,3), 0, dtype=int)
        grid_one_cell_not_fill[1,1] = 1
        unpadded_one_cell = unpad_grid(grid_one_cell_not_fill, fill_val=0)
        self.assertEqual(unpadded_one_cell.shape, (1,1))
        np.testing.assert_array_equal(unpadded_one_cell, np.array([[1]]))


if __name__ == '__main__':
    unittest.main()
