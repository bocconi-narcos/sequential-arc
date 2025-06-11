import unittest
import numpy as np

# Import DSL modules
from dsl.colour import ColorSelector
from dsl.select import GridSelector
from dsl.transform import GridTransformer
from dsl.utils.background import find_background_colour
from dsl.utils.padding import pad_grid, unpad_grid

class TestGeneralDSL(unittest.TestCase):

    def setUp(self):
        self.grid1 = np.array([[1, 2, 1], [0, 1, 2], [2, 0, 0]]) # 3x3, multiple colors
        self.grid2 = np.array([[5, 5, 5], [5, 0, 5], [5, 5, 5]]) # 3x3, mostly one color
        self.grid3 = np.array([[0]]) # 1x1 grid
        self.grid4 = np.array([[1, 1], [1, 1]]) # 2x2, all one color
        self.grid5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 3x3, all different colors
        self.grid_empty = np.array([[]], dtype=int) # Empty grid (0 rows, 0 cols conceptually)
        self.grid_empty_actual = np.empty((0,0), dtype=int) # Actual 0x0 grid

        self.selection1 = np.array([[True, False, True], [False, True, False], [True, False, False]])
        self.selection_all_true = np.ones_like(self.grid1, dtype=bool)
        self.selection_all_false = np.zeros_like(self.grid1, dtype=bool)
        self.grid_selector_instance = GridSelector(min_geometry=1) # For rectangles

    # --- Helper Assertions ---
    def assertColorInRange(self, color):
        self.assertIsInstance(color, (int, np.integer))
        self.assertTrue(0 <= color <= 9, f"Color {color} out of range 0-9")

    def assertMaskBasic(self, mask, expected_shape, expected_dims=2):
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.ndim, expected_dims)
        self.assertEqual(mask.shape, expected_shape)
        self.assertTrue(np.all(np.isin(mask, [False, True])))

    def assertGridBasic(self, output_grid, input_grid, is_new_object=True):
        self.assertIsInstance(output_grid, np.ndarray)
        self.assertEqual(output_grid.ndim, 2)
        self.assertEqual(output_grid.shape, input_grid.shape)
        if is_new_object:
            self.assertIsNot(output_grid, input_grid)
        # Check if colors are generally valid (0-9 or specified fill values)
        # This is a loose check, specific transformations should check values more precisely
        if output_grid.size > 0:
             self.assertTrue(np.all((output_grid >= 0) & (output_grid <= 9)) or np.any(output_grid == -1),
                            f"Grid contains invalid color values: {output_grid}")


    # --- Test Color Output Ranges (ColorSelector) ---
    def test_color_selector_output_range(self):
        grids_to_test = [self.grid1, self.grid2, self.grid3, self.grid4, self.grid5]
        color_selector_fns_simple = [
            ColorSelector.most_common,
            ColorSelector.least_common,
        ]
        color_selector_fns_with_color_arg = [
            (ColorSelector.colour, {"colour": 0}), (ColorSelector.colour_0, {}),
            (ColorSelector.colour_1, {}), (ColorSelector.colour_2, {}),
            (ColorSelector.colour_3, {}), (ColorSelector.colour_4, {}),
            (ColorSelector.colour_5, {}), (ColorSelector.colour_6, {}),
            (ColorSelector.colour_7, {}), (ColorSelector.colour_8, {}),
            (ColorSelector.colour_9, {}),
        ]
        color_selector_fns_with_rank_arg = [
            (ColorSelector.nth_most_common, {"rank": 0}),
            (ColorSelector.nth_most_common, {"rank": 1}),
            (ColorSelector.second_most_common, {}),
            (ColorSelector.nth_most_independent, {"rank": 0, "connectivity": 4}),
            (ColorSelector.most_independent_cells, {}),
            (ColorSelector.colour_of_nth_largest_shape, {"rank": 0}),
        ]

        for grid_idx, grid in enumerate(grids_to_test):
            if grid.size == 0: continue # Skip empty for these, handled in invalid tests

            for func in color_selector_fns_simple:
                with self.subTest(func=func.__name__, grid_idx=grid_idx):
                    color = func(grid)
                    self.assertColorInRange(color)

            for func, kwargs in color_selector_fns_with_color_arg:
                 with self.subTest(func=func.__name__, grid_idx=grid_idx, kwargs=kwargs):
                    # The 'colour' function itself returns the input color if valid, not a selection from grid
                    if func.__name__ == 'colour':
                        color_param = kwargs.get("colour", 0)
                        output_color = func(grid, **kwargs)
                        self.assertEqual(output_color, color_param)
                        self.assertColorInRange(output_color)
                    elif f"colour_{kwargs.get('colour', func.__name__.split('_')[-1])}" == func.__name__ : # for colour_0 etc.
                        expected_c = int(func.__name__.split('_')[-1])
                        output_color = func(grid, **kwargs)
                        self.assertEqual(output_color, expected_c)
                        self.assertColorInRange(output_color)


            for func, kwargs in color_selector_fns_with_rank_arg:
                # Adjust rank for smaller grids to avoid errors, or ensure errors are handled if that's the test
                current_kwargs = kwargs.copy()
                if "rank" in current_kwargs:
                    # Ensure rank is valid for the grid size/content or expect graceful handling
                    # For simplicity here, we assume rank 0/1 should usually work. More specific tests per func needed for edge ranks.
                    if grid.size < 5 and current_kwargs["rank"] > 0 : # simplified check
                         current_kwargs["rank"] = 0

                with self.subTest(func=func.__name__, grid_idx=grid_idx, kwargs=current_kwargs):
                    color = func(grid, **current_kwargs)
                    self.assertColorInRange(color)

        # Test specific cases
        self.assertEqual(ColorSelector.most_common(self.grid1), 0) # 0 appears 3 times
        self.assertEqual(ColorSelector.least_common(self.grid1), 1) # 1 appears 2 times (0:3, 1:2, 2:2) - tie break? -> 1
        self.assertEqual(ColorSelector.nth_most_common(self.grid1, rank=0), 0)
        self.assertEqual(ColorSelector.nth_most_common(self.grid1, rank=1), 1) # or 2, depends on tie-breaking for count
        self.assertEqual(ColorSelector.colour_of_nth_largest_shape(np.array([[1,1,0],[1,0,0]]), rank=0), 1)


    # --- Test Selection Output Types and Dimensions (GridSelector) ---
    def test_grid_selector_output_type_and_dims(self):
        grids_to_test = [self.grid1, self.grid3, self.grid4] # grid2, grid5 are similar enough for basic type checks

        # Functions returning a single 2D boolean mask
        selector_fns_2d = [
            (GridSelector.all_cells, {}),
            (GridSelector.colour, {"colour": 1}),
            (GridSelector.nth_largest_shape, {"colour": 1, "connectivity": 4, "rank": 0}),
            (GridSelector.independent_cells, {"colour": 0, "connectivity": 4}),
            (GridSelector.independent_cells_4, {"colour": 0}),
            (GridSelector.grid_border, {"colour": 0}), # colour arg might be ignored by some
            (GridSelector.adjacent4, {"colour": 1, "contacts": 1}),
            (GridSelector.contact4_1, {"colour": 1}),
            (GridSelector.adjacent8, {"colour": 1, "contacts": 1}),
        ]

        # Functions returning a stack of 3D boolean masks (N, H, W)
        selector_fns_3d = [
            (GridSelector.components4, {"colour": 1}),
            (GridSelector.components8, {"colour": 1}),
            (GridSelector.outer_border4, {"colour": 1}),
            (GridSelector.inner_border4, {"colour": 1}),
            (GridSelector.outer_border8, {"colour": 1}),
            (GridSelector.outer_border_8, {"colour": 1}),
            (GridSelector.inner_border8, {"colour": 1}),
            (self.grid_selector_instance.rectangles, {"colour": 1, "height": 1, "width": 1})
        ]

        for grid_idx, grid in enumerate(grids_to_test):
            if grid.size == 0: continue

            for func, kwargs in selector_fns_2d:
                current_kwargs = kwargs.copy()
                if "colour" in current_kwargs and np.sum(grid == current_kwargs["colour"]) == 0 and func.__name__ == "nth_largest_shape":
                    # If color for nth_largest_shape is not present, it should return all_false
                    pass # Expected to return empty/all_false mask

                with self.subTest(func=func.__name__, grid_idx=grid_idx, type="2D"):
                    mask = func(grid, **current_kwargs)
                    self.assertMaskBasic(mask, grid.shape, expected_dims=2)
                    if func.__name__ == "all_cells":
                        self.assertTrue(np.all(mask))

            for func, kwargs in selector_fns_3d:
                current_kwargs = kwargs.copy()
                # Adjust for small grids if necessary, e.g. for rectangles
                if func.__name__ == "rectangles":
                    current_kwargs["height"] = min(current_kwargs["height"], grid.shape[0])
                    current_kwargs["width"] = min(current_kwargs["width"], grid.shape[1])

                # Skip if color for components is not present, as it returns (0,H,W)
                if "colour" in current_kwargs and np.sum(grid == current_kwargs["colour"]) == 0 and \
                   func.__name__ in ["components4", "components8", "outer_border4", "inner_border4", "outer_border8", "inner_border8"]:
                    with self.subTest(func=func.__name__, grid_idx=grid_idx, type="3D_no_components"):
                        mask_stack = func(grid, **current_kwargs)
                        self.assertMaskBasic(mask_stack, (0, grid.shape[0], grid.shape[1]), expected_dims=3)
                    continue

                with self.subTest(func=func.__name__, grid_idx=grid_idx, type="3D"):
                    mask_stack = func(grid, **current_kwargs)
                    self.assertIsInstance(mask_stack, np.ndarray)
                    self.assertEqual(mask_stack.dtype, bool)
                    self.assertEqual(mask_stack.ndim, 3)
                    self.assertTrue(mask_stack.shape[0] >= 0) # Can be 0 if no components/rects found
                    self.assertEqual(mask_stack.shape[1:], grid.shape if mask_stack.shape[0] > 0 else (grid.shape[0],grid.shape[1]))
                    self.assertTrue(np.all(np.isin(mask_stack, [False, True])))

        # Specific check for nth_largest_shape when no component of color
        mask_no_comp = GridSelector.nth_largest_shape(self.grid1, colour=9, connectivity=4, rank=0) # Color 9 not in grid1
        self.assertMaskBasic(mask_no_comp, self.grid1.shape)
        self.assertFalse(np.any(mask_no_comp))


    # --- Test Transformation Output (GridTransformer) ---
    def test_grid_transformer_output(self):
        grids_to_test = [self.grid1, self.grid4] # grid3 is too small for some ops like rotate non-180
        selections_to_test = [self.selection1, self.selection_all_true]

        transformer_fns = [
            (GridTransformer.identity, {}),
            (GridTransformer.clear, {}),
            (GridTransformer.new_colour, {"color": 5}),
            (GridTransformer.new_colour_5, {}),
            (GridTransformer.background_colour, {}),
            (GridTransformer.invert_colors, {}),
            (GridTransformer.flip, {"axis": 0}), (GridTransformer.flip_vertical, {}),
            (GridTransformer.flip, {"axis": 1}), (GridTransformer.flip_horizontal, {}),
            (GridTransformer.rotate, {"k": 1}), (GridTransformer.rotate_90, {}),
            (GridTransformer.rotate, {"k": 2}), (GridTransformer.rotate_180, {}),
            (GridTransformer.rotate, {"k": 3}), (GridTransformer.rotate_270, {}),
        ]
        # Slide functions are more complex and tested separately due to many params
        slide_new_params_sets = [
            {"direction": "down", "mode": "copy", "continuous": False, "obstacles": True, "fluid": False, "superfluid": False},
            {"direction": "up", "mode": "cut", "continuous": False, "obstacles": True, "fluid": True, "superfluid": False},
            {"direction": "left", "mode": "copy", "continuous": True, "obstacles": False, "fluid": False, "superfluid": False}, # continuous not with fluid
            {"direction": "right", "mode": "cut", "continuous": False, "obstacles": True, "fluid": True, "superfluid": True},
        ]
        slide_old_params_sets = [ # Similar to slide_new, focus on key differences if any
            {"direction": "down", "mode": "cut", "continuous": False, "fluid": True, "superfluid": True}, # move_down_superfluid
            {"direction": "right", "mode": "copy", "continuous": False, "obstacles": False, "fluid": False},
        ]


        for grid_idx, grid in enumerate(grids_to_test):
            for sel_idx, selection_orig in enumerate(selections_to_test):
                # Ensure selection matches grid shape if testing with grid1's selection
                selection = selection_orig if grid.shape == selection_orig.shape else np.ones_like(grid, dtype=bool)
                if grid.size == 0: continue

                for func, kwargs in transformer_fns:
                    with self.subTest(func=func.__name__, grid_idx=grid_idx, sel_idx=sel_idx, kwargs=kwargs):
                        transformed_grid = func(grid, selection, **kwargs)
                        self.assertGridBasic(transformed_grid, grid)
                        if func.__name__ == "clear" and np.any(selection):
                             self.assertTrue(np.all(transformed_grid[selection] == 0))
                        if func.__name__ == "new_colour" and np.any(selection):
                            self.assertTrue(np.all(transformed_grid[selection] == kwargs["color"]))

                for slide_func, params_sets in [(GridTransformer.slide_new, slide_new_params_sets),
                                                (GridTransformer.slide_old, slide_old_params_sets)]:
                    for s_idx, params in enumerate(params_sets):
                        # Skip invalid combos for slide_new
                        if slide_func.__name__ == "slide_new" and params["continuous"] and (params["fluid"] or params["superfluid"]):
                            continue
                        if slide_func.__name__ == "slide_new" and params["superfluid"] and not params["fluid"]:
                            continue

                        with self.subTest(func=slide_func.__name__, params_idx=s_idx, grid_idx=grid_idx, sel_idx=sel_idx):
                            transformed_grid = slide_func(grid, selection, **params)
                            self.assertGridBasic(transformed_grid, grid)

    # --- Test Utils Functions ---
    def test_utils_functions(self):
        # find_background_colour
        self.assertEqual(find_background_colour(self.grid1), 0)
        self.assertEqual(find_background_colour(self.grid2), 5)
        self.assertEqual(find_background_colour(self.grid3), 0)
        with self.assertRaises(ValueError): # Expect error on non-2D
            find_background_colour(np.array([1,2,3]))
        # find_background_colour on empty grid might be undefined or error, check docs.
        # Based on np.unique, it might error on fully empty, or return garbage if not checked.
        # Assuming it should handle grid_empty (0,N) or (N,0) gracefully or error.
        # Current impl will error due to argmax on empty array from np.unique.
        with self.assertRaises(ValueError): # Or IndexError depending on numpy version for argmax
            find_background_colour(self.grid_empty_actual)


        # pad_grid
        padded1 = pad_grid(self.grid1, (5,5), fill_val=-1)
        self.assertGridBasic(padded1, np.empty((5,5)), is_new_object=True) # Shape check against target
        self.assertEqual(padded1.shape, (5,5))
        self.assertTrue(np.all(padded1[:3,:3] == self.grid1))
        self.assertTrue(np.all(padded1[3:,:] == -1) or np.all(padded1[:,3:] == -1)) # Check fill
        with self.assertRaises(ValueError): # Grid larger than canvas
            pad_grid(self.grid1, (2,2))

        # unpad_grid
        grid_to_unpad = pad_grid(self.grid1, (5,5), fill_val=0) # Pad with 0
        unpadded1 = unpad_grid(grid_to_unpad, fill_val=0)
        self.assertGridBasic(unpadded1, self.grid1, is_new_object=False) # May return view or copy
        self.assertTrue(np.array_equal(unpadded1, self.grid1))

        all_fill = np.full((3,3), 0)
        unpadded_empty = unpad_grid(all_fill, fill_val=0)
        self.assertEqual(unpadded_empty.shape, (0,0)) # Should be (0,0) or (0,X) or (X,0)

        unpadded_original = unpad_grid(self.grid1, fill_val=-5) # Fill val not present
        self.assertTrue(np.array_equal(unpadded_original, self.grid1))


    # --- Test Handling of Empty/Invalid Grids ---
    def test_invalid_grid_handling(self):
        invalid_grids = [
            None,
            [[1,2],[3,4]], # list of lists
            np.array([1,2,3]), # 1D
            np.array([[[1],[2]],[[3],[4]]]), # 3D
            self.grid_empty, # (1,0) or (0,1) shape, effectively empty cells
            self.grid_empty_actual # (0,0) shape
        ]
        # Selection for transformers, needs to be valid for the functions that need it
        # For invalid grid tests, selection validity is secondary or also part of test
        dummy_selection = np.array([[True]])

        color_selectors = [
            ColorSelector.most_common,
            (ColorSelector.colour, {"colour": 0}),
            (ColorSelector.nth_most_common, {"rank": 0})
        ]
        grid_selectors = [
            GridSelector.all_cells,
            (GridSelector.colour, {"colour": 0}),
            (GridSelector.components4, {"colour": 0})
        ]
        grid_transformers = [
            GridTransformer.identity,
            (GridTransformer.clear, {}), # Requires selection
            (GridTransformer.flip, {"axis":0}) # Requires selection
        ]

        for grid_idx, invalid_grid in enumerate(invalid_grids):
            for func_item in color_selectors:
                func, kwargs = (func_item, {}) if callable(func_item) else func_item
                with self.subTest(type="ColorSelector", func=func.__name__, grid_idx=grid_idx):
                    with self.assertRaises(ValueError): # Most should raise ValueError for bad grid
                        func(invalid_grid, **kwargs)

            for func_item in grid_selectors:
                func, kwargs = (func_item, {}) if callable(func_item) else func_item
                with self.subTest(type="GridSelector", func=func.__name__, grid_idx=grid_idx):
                     # all_cells might handle np.array([[]]) differently (makes 1,0 bool array)
                    if func == GridSelector.all_cells and isinstance(invalid_grid, np.ndarray) and invalid_grid.size==0 :
                        try:
                            res = func(invalid_grid, **kwargs)
                            self.assertMaskBasic(res, invalid_grid.shape) # e.g. (0,0) or (1,0)
                        except ValueError:
                            pass # Or it might raise ValueError, also acceptable
                    else:
                        with self.assertRaises(ValueError):
                            func(invalid_grid, **kwargs)

            for func_item in grid_transformers:
                func, kwargs_ = (func_item, {}) if callable(func_item) else func_item
                # Transformers need a selection. We'll use a dummy one or make one that matches shape if possible.
                current_selection = dummy_selection
                if isinstance(invalid_grid, np.ndarray) and invalid_grid.ndim == 2:
                    current_selection = np.ones(invalid_grid.shape, dtype=bool) if invalid_grid.size > 0 else np.empty(invalid_grid.shape, dtype=bool)

                with self.subTest(type="GridTransformer", func=func.__name__, grid_idx=grid_idx):
                    if func == GridTransformer.identity and isinstance(invalid_grid, np.ndarray) and invalid_grid.ndim == 2:
                        # Identity might be more lenient if it just copies
                        try:
                            res = func(invalid_grid, current_selection, **kwargs_)
                            self.assertGridBasic(res, invalid_grid, is_new_object=True)
                        except ValueError: # Or raise an error
                            pass
                    else:
                        with self.assertRaises(ValueError): # Expect ValueError due to bad grid or mask mismatch
                            func(invalid_grid, current_selection, **kwargs_)

        # Test empty actual grid (0,0) specifically where it might not be a ValueError but return empty structures
        grid_0x0 = self.grid_empty_actual
        self.assertEqual(ColorSelector.most_common(grid_0x0), 0) # bincount on empty gives [0], argmax is 0
        self.assertEqual(ColorSelector.least_common(grid_0x0), 0) # minlength pushes it, argmin is 0

        mask_0x0_all = GridSelector.all_cells(grid_0x0)
        self.assertMaskBasic(mask_0x0_all, (0,0))

        mask_0x0_color = GridSelector.colour(grid_0x0, colour=1)
        self.assertMaskBasic(mask_0x0_color, (0,0))

        comp_0x0 = GridSelector.components4(grid_0x0, colour=1)
        self.assertMaskBasic(comp_0x0, (0,0,0), expected_dims=3)

        transformed_0x0 = GridTransformer.identity(grid_0x0, np.empty((0,0), dtype=bool))
        self.assertGridBasic(transformed_0x0, grid_0x0)

        transformed_0x0_clear = GridTransformer.clear(grid_0x0, np.empty((0,0), dtype=bool))
        self.assertGridBasic(transformed_0x0_clear, grid_0x0)

        # Test with invalid selection masks for transformers
        with self.assertRaises(ValueError):
            GridTransformer.clear(self.grid1, np.array([[True, False], [False, True]])) # Shape mismatch
        with self.assertRaises(ValueError):
            GridTransformer.clear(self.grid1, self.grid1) # Wrong dtype for mask

if __name__ == '__main__':
    unittest.main()
