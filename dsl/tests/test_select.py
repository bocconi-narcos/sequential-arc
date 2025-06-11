import unittest
import numpy as np
from dsl.select import GridSelector
from dsl.utils.background import find_background_colour

class TestGridSelectorSpecific(unittest.TestCase):

    def setUp(self):
        self.selector = GridSelector(min_geometry=1) # min_geometry=1 for easier testing of small rectangles
        self.grid_empty = np.empty((0,0), dtype=int)
        self.grid_1x1_c0 = np.array([[0]])
        self.grid_3x3_standard = np.array([[1, 1, 0], [1, 0, 2], [0, 2, 2]])
        # Components for grid_3x3_standard:
        # Color 1 (4-conn): [[T,T,F],[T,F,F],[F,F,F]]
        # Color 0 (4-conn): [[F,F,T],[F,T,F],[T,F,F]] (two components)
        # Color 2 (4-conn): [[F,F,F],[F,F,T],[F,T,T]]
        self.grid_hollow_frame = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ])
        self.grid_all_color1 = np.ones((3,3), dtype=int)
        self.grid_for_rects = np.array([
            [1,1,0,1,1],
            [1,1,0,1,1],
            [0,0,0,0,0],
            [2,2,2,0,0],
            [2,2,2,0,0]
        ])

    def assertMaskBasic(self, mask, expected_shape, expected_dims=2):
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool, f"Mask dtype is not bool: {mask.dtype}")
        self.assertEqual(mask.ndim, expected_dims, f"Mask ndim is not {expected_dims}: {mask.ndim}")
        self.assertEqual(mask.shape, expected_shape, f"Mask shape is not {expected_shape}: {mask.shape}")
        if mask.size > 0: # Only check values if mask is not empty
            self.assertTrue(np.all(np.isin(mask, [False, True])), "Mask contains values other than True/False")

    # --- Test GridSelector.all_cells ---
    def test_all_cells_specific(self):
        grids_to_test = [self.grid_1x1_c0, self.grid_3x3_standard, self.grid_empty, np.empty((2,0), dtype=int)]
        for i, grid in enumerate(grids_to_test):
            with self.subTest(grid_idx=i):
                mask = GridSelector.all_cells(grid)
                expected_mask = np.ones_like(grid, dtype=bool)
                self.assertMaskBasic(mask, grid.shape)
                np.testing.assert_array_equal(mask, expected_mask)

    # --- Test GridSelector.colour ---
    def test_colour_specific(self):
        mask_has_color = GridSelector.colour(self.grid_3x3_standard, colour=1)
        expected_mask_has_color = np.array([[True,True,False],[True,False,False],[False,False,False]])
        self.assertMaskBasic(mask_has_color, self.grid_3x3_standard.shape)
        np.testing.assert_array_equal(mask_has_color, expected_mask_has_color)

        mask_no_color = GridSelector.colour(self.grid_3x3_standard, colour=5)
        expected_mask_no_color = np.zeros_like(self.grid_3x3_standard, dtype=bool)
        self.assertMaskBasic(mask_no_color, self.grid_3x3_standard.shape)
        np.testing.assert_array_equal(mask_no_color, expected_mask_no_color)

        mask_empty = GridSelector.colour(self.grid_empty, colour=1)
        self.assertMaskBasic(mask_empty, self.grid_empty.shape)

        with self.assertRaisesRegex(ValueError, "colour must be a non‑negative integer"):
            GridSelector.colour(self.grid_3x3_standard, colour=-1)
        with self.assertRaisesRegex(ValueError, "colour must be a non‑negative integer"):
            GridSelector.colour(self.grid_3x3_standard, colour=1.5)


    # --- Test GridSelector.components4 and components8 ---
    def test_components_specific(self):
        # Test components4
        comps4_c1 = GridSelector.components4(self.grid_3x3_standard, colour=1)
        self.assertMaskBasic(comps4_c1, (1, 3, 3), expected_dims=3)
        np.testing.assert_array_equal(comps4_c1[0], [[True,True,False],[True,False,False],[False,False,False]])

        comps4_c0 = GridSelector.components4(self.grid_3x3_standard, colour=0)
        self.assertMaskBasic(comps4_c0, (2, 3, 3), expected_dims=3) # Two components for color 0
        # Could check individual components if order is guaranteed (usually by scanline)

        comps4_c5 = GridSelector.components4(self.grid_3x3_standard, colour=5) # Color not present
        self.assertMaskBasic(comps4_c5, (0, 3, 3), expected_dims=3)

        comps4_all_color1 = GridSelector.components4(self.grid_all_color1, colour=1)
        self.assertMaskBasic(comps4_all_color1, (1, 3, 3), expected_dims=3)
        np.testing.assert_array_equal(comps4_all_color1[0], np.ones((3,3), dtype=bool))

        comps4_empty = GridSelector.components4(self.grid_empty, colour=1)
        self.assertMaskBasic(comps4_empty, (0,0,0), expected_dims=3) # Shape (0, H, W)

        # Test components8 (example with diagonal connection)
        grid_diag = np.array([[1,0,0],[0,1,0],[0,0,1]])
        comps8_c1_diag = GridSelector.components8(grid_diag, colour=1)
        self.assertMaskBasic(comps8_c1_diag, (1, 3, 3), expected_dims=3)
        np.testing.assert_array_equal(comps8_c1_diag[0], [[True,False,False],[False,True,False],[False,False,True]])

        comps4_c1_diag = GridSelector.components4(grid_diag, colour=1) # Should be 3 components
        self.assertMaskBasic(comps4_c1_diag, (3, 3, 3), expected_dims=3)


    # --- Test GridSelector.nth_largest_shape ---
    def test_nth_largest_shape_specific(self):
        grid = np.array([
            [1,1,0,2,2,2], # C1 (size 2), C2 (size 3)
            [1,0,0,0,0,0],
            [3,3,3,3,0,4], # C3 (size 4), C4 (size 1)
            [0,0,0,0,0,4]  # C4 (another size 1, total size 2 for color 4)
        ])
        # Color 1: size 2 at (0,0)
        # Color 2: size 3 at (0,3)
        # Color 3: size 4 at (2,0)
        # Color 4: size 2 (from two components of size 1) - careful, components are distinct

        # Test for color 3 (largest single component)
        mask_c3_r0 = GridSelector.nth_largest_shape(grid, colour=3, connectivity=4, rank=0)
        np.testing.assert_array_equal(mask_c3_r0, [[F,F,F,F,F,F],[F,F,F,F,F,F],[T,T,T,T,F,F],[F,F,F,F,F,F]], T=True,F=False)

        # Test for color 2 (second largest single component)
        mask_c2_r0 = GridSelector.nth_largest_shape(grid, colour=2, connectivity=4, rank=0)
        np.testing.assert_array_equal(mask_c2_r0, [[F,F,F,T,T,T],[F,F,F,F,F,F],[F,F,F,F,F,F],[F,F,F,F,F,F]], T=True,F=False)

        # Test for color 1 (third largest single component)
        mask_c1_r0 = GridSelector.nth_largest_shape(grid, colour=1, connectivity=4, rank=0)
        np.testing.assert_array_equal(mask_c1_r0, [[T,T,F,F,F,F],[T,F,F,F,F,F],[F,F,F,F,F,F],[F,F,F,F,F,F]], T=True,F=False)

        # Test for color 4. It has two components of size 1.
        # Comp1: (2,5), Comp2: (3,5). Comp1 comes first by tie-breaking.
        mask_c4_r0 = GridSelector.nth_largest_shape(grid, colour=4, connectivity=4, rank=0)
        np.testing.assert_array_equal(mask_c4_r0, [[F,F,F,F,F,F],[F,F,F,F,F,F],[F,F,F,F,F,T],[F,F,F,F,F,F]], T=True,F=False)
        mask_c4_r1 = GridSelector.nth_largest_shape(grid, colour=4, connectivity=4, rank=1)
        np.testing.assert_array_equal(mask_c4_r1, [[F,F,F,F,F,F],[F,F,F,F,F,F],[F,F,F,F,F,F],[F,F,F,F,F,T]], T=True,F=False)

        # Rank out of bounds
        mask_c1_r_oob = GridSelector.nth_largest_shape(grid, colour=1, connectivity=4, rank=1)
        self.assertFalse(np.any(mask_c1_r_oob)) # Should be all False

        # Color not present
        mask_c5_r0 = GridSelector.nth_largest_shape(grid, colour=5, connectivity=4, rank=0)
        self.assertFalse(np.any(mask_c5_r0))

        with self.assertRaisesRegex(ValueError, "connectivity must be 4 or 8"):
            GridSelector.nth_largest_shape(grid, colour=1, connectivity=5, rank=0)

        # Tie-breaking example from problem spec
        grid_tie = np.array([[1,0,1],[1,0,1],[0,0,0]])
        mask_tie_r0 = GridSelector.nth_largest_shape(grid_tie, colour=1, connectivity=4, rank=0)
        np.testing.assert_array_equal(mask_tie_r0, [[T,F,F],[T,F,F],[F,F,F]], T=True,F=False)
        mask_tie_r1 = GridSelector.nth_largest_shape(grid_tie, colour=1, connectivity=4, rank=1)
        np.testing.assert_array_equal(mask_tie_r1, [[F,F,T],[F,F,T],[F,F,F]], T=True,F=False)


    # --- Test GridSelector.independent_cells ---
    def test_independent_cells_specific(self):
        grid = np.array([ # Background is 0
            [1, 0, 2, 0, 1], # 1s are independent. 2 is independent.
            [0, 3, 3, 0, 0], # 3s are not independent.
            [4, 0, 5, 5, 0], # 4 is independent. 5s are not.
            [0, 1, 0, 6, 0]  # This 1 is independent. 6 is independent.
        ])
        # Expected for color 1 (connectivity 4): (0,0), (0,4), (3,1)
        mask_c1 = GridSelector.independent_cells(grid, colour=1, connectivity=4)
        expected_c1 = np.array([
            [T,F,F,F,T],
            [F,F,F,F,F],
            [F,F,F,F,F],
            [F,T,F,F,F]
        ], dtype=bool, T=True)
        np.testing.assert_array_equal(mask_c1, expected_c1)

        # Test partial
        mask_c1_p = GridSelector.independent_cells_4(grid, colour=1)
        np.testing.assert_array_equal(mask_c1_p, expected_c1)

        # Test color 2
        mask_c2 = GridSelector.independent_cells(grid, colour=2, connectivity=4)
        expected_c2 = np.array([[F,F,T,F,F],[F,F,F,F,F],[F,F,F,F,F],[F,F,F,F,F]], dtype=bool, T=True)
        np.testing.assert_array_equal(mask_c2, expected_c2)

        # Test color 3 (not independent)
        mask_c3 = GridSelector.independent_cells(grid, colour=3, connectivity=4)
        self.assertFalse(np.any(mask_c3))

        # Test with color being background
        mask_c0 = GridSelector.independent_cells(grid, colour=0, connectivity=4) # Background cannot be independent
        self.assertFalse(np.any(mask_c0))

        # Test all same color (no independent cells if not background)
        mask_all1 = GridSelector.independent_cells(self.grid_all_color1, colour=1, connectivity=4)
        self.assertFalse(np.any(mask_all1))

    # --- Test border functions ---
    def test_borders_specific(self):
        # Outer borders
        outer4_hollow = GridSelector.outer_border4(self.grid_hollow_frame, colour=1)
        self.assertEqual(outer4_hollow.shape[0], 1) # One component of color 1
        expected_outer_hollow = np.array([
            [T,T,T,T],
            [T,F,F,T],
            [T,F,F,T],
            [T,T,T,T]], dtype=bool, T=True)
        np.testing.assert_array_equal(outer4_hollow[0], find_boundaries(self.grid_hollow_frame==1, mode="outer", connectivity=1)) # skimage conn=1 for 4-way

        outer8_hollow = GridSelector.outer_border8(self.grid_hollow_frame, colour=1) # Same for this shape
        np.testing.assert_array_equal(outer8_hollow[0], find_boundaries(self.grid_hollow_frame==1, mode="outer", connectivity=2)) # skimage conn=2 for 8-way

        # Inner borders
        inner4_hollow = GridSelector.inner_border4(self.grid_hollow_frame, colour=1)
        self.assertEqual(inner4_hollow.shape[0], 1)
        expected_inner_hollow = np.array([
            [F,F,F,F],
            [F,T,T,F],
            [F,T,T,F],
            [F,F,F,F]], dtype=bool, T=True)
        np.testing.assert_array_equal(inner4_hollow[0], find_boundaries(self.grid_hollow_frame==1, mode="inner", connectivity=1))

        inner4_solid = GridSelector.inner_border4(self.grid_all_color1, colour=1) # Solid shape
        self.assertEqual(inner4_solid.shape[0],1)
        self.assertFalse(np.any(inner4_solid[0])) # No inner border

        # Grid border
        border_mask = GridSelector.grid_border(self.grid_3x3_standard, colour=99) # color arg is ignored
        expected_gb = np.array([[T,T,T],[T,F,T],[T,T,T]], dtype=bool,T=True)
        np.testing.assert_array_equal(border_mask, expected_gb)
        border_mask_1x1 = GridSelector.grid_border(self.grid_1x1_c0, colour=0)
        np.testing.assert_array_equal(border_mask_1x1, [[True]])
        border_mask_empty = GridSelector.grid_border(self.grid_empty, colour=0)
        self.assertMaskBasic(border_mask_empty, self.grid_empty.shape)


    # --- Test adjacency functions ---
    def test_adjacency_specific(self):
        grid = np.array([
            [0,1,0],
            [1,2,1], # Color 2 is central, surrounded by 1s
            [0,1,0]
        ])
        # Adjacent4 to color 2
        adj4_c2_k1 = GridSelector.adjacent4(grid, colour=2, contacts=1)
        # Cells (0,1), (1,0), (1,2), (2,1) are color 1, each touching color 2 once.
        expected_adj4_c2_k1 = np.array([[F,T,F],[T,F,T],[F,T,F]], dtype=bool,T=True)
        np.testing.assert_array_equal(adj4_c2_k1, expected_adj4_c2_k1)

        adj4_c2_k0 = GridSelector.adjacent4(grid, colour=2, contacts=0) # contacts=0 means no contact
        # This is not handled by the function's explicit check (1-4), relies on convolve result
        # The current logic `hit = np.any(counts == contacts, axis=0)` will look for 0 contacts
        # This means cells not touching the component.
        # It's probably better to test contacts 1-4 as per error check.
        with self.assertRaisesRegex(ValueError, "contacts must be 1–4"):
             GridSelector.adjacent4(grid, colour=2, contacts=0)
        with self.assertRaisesRegex(ValueError, "contacts must be 1–4"):
             GridSelector.adjacent4(grid, colour=2, contacts=5)

        # Adjacent8 to color 2 (all 0s and 1s around it)
        grid_diag_adj = np.array([
            [1,1,1],
            [1,2,1],
            [1,1,1]
        ])
        adj8_c2_k8 = GridSelector.adjacent8(grid_diag_adj, colour=2, contacts=8)
        expected_adj8_c2_k8 = np.array([[T,T,T],[T,F,T],[T,T,T]], dtype=bool,T=True)
        np.testing.assert_array_equal(adj8_c2_k8, expected_adj8_c2_k8)

        with self.assertRaisesRegex(ValueError, "contacts must be 1–8"):
            GridSelector.adjacent8(grid_diag_adj, colour=2, contacts=0)
        with self.assertRaisesRegex(ValueError, "contacts must be 1–8"):
            GridSelector.adjacent8(grid_diag_adj, colour=2, contacts=9)

        # Test partials
        np.testing.assert_array_equal(GridSelector.contact4_1(grid,colour=2), GridSelector.adjacent4(grid,colour=2,contacts=1))
        np.testing.assert_array_equal(GridSelector.contact8_1(grid_diag_adj,colour=2), GridSelector.adjacent8(grid_diag_adj,colour=2,contacts=1))


    # --- Test GridSelector.rectangles ---
    def test_rectangles_specific(self):
        # Test with self.grid_for_rects
        # Color 1, 2x2 rects (min_geometry=1 for self.selector)
        rects_c1_2x2 = self.selector.rectangles(self.grid_for_rects, colour=1, height=2, width=2)
        self.assertEqual(rects_c1_2x2.shape[0], 2) # Two 2x2 of color 1
        expected_r1 = np.zeros_like(self.grid_for_rects, dtype=bool)
        expected_r1[0:2,0:2] = True
        expected_r2 = np.zeros_like(self.grid_for_rects, dtype=bool)
        expected_r2[0:2,3:5] = True
        # Order of rects might vary, check if both are present
        self.assertTrue(np.any([np.array_equal(rects_c1_2x2[i], expected_r1) for i in range(2)]))
        self.assertTrue(np.any([np.array_equal(rects_c1_2x2[i], expected_r2) for i in range(2)]))

        # Color 2, 2x3 rects
        rects_c2_2x3 = self.selector.rectangles(self.grid_for_rects, colour=2, height=2, width=3)
        self.assertEqual(rects_c2_2x3.shape[0], 1)
        expected_r_c2 = np.zeros_like(self.grid_for_rects, dtype=bool)
        expected_r_c2[3:5,0:3] = True
        np.testing.assert_array_equal(rects_c2_2x3[0], expected_r_c2)

        # No such rectangles
        rects_none = self.selector.rectangles(self.grid_for_rects, colour=1, height=3, width=3)
        self.assertEqual(rects_none.shape[0], 0)

        # Height/width too small (if min_geometry was > 1) or too large
        # self.selector has min_geometry=1, so h=1, w=1 is fine
        rects_1x1 = self.selector.rectangles(self.grid_1x1_c0, colour=0, height=1, width=1)
        self.assertEqual(rects_1x1.shape[0], 1)

        gs_min2 = GridSelector(min_geometry=2)
        rects_min2_fail = gs_min2.rectangles(self.grid_1x1_c0, colour=0, height=1, width=1)
        self.assertEqual(rects_min2_fail.shape[0], 0) # height < min_geometry

        rects_too_large = self.selector.rectangles(self.grid_for_rects, colour=1, height=10, width=10)
        self.assertEqual(rects_too_large.shape[0], 0)


    # --- Input Validation (specific call paths) ---
    def test_input_validation_specific(self):
        invalid_grids = [
            ([1,2,3]),
            (np.array([1,2,3])),
            (np.array([[[1]]])),
            (np.array([["a"]]))
        ]
        # Functions and their specific args for a basic valid call (excluding grid)
        # Some functions might not take a color arg (e.g. all_cells)
        # Some functions have more required args (e.g. nth_largest_shape)
        test_configs = [
            (GridSelector.all_cells, {}),
            (GridSelector.colour, {"colour": 0}),
            (GridSelector.components4, {"colour": 0}),
            (GridSelector.components8, {"colour": 0}),
            (GridSelector.nth_largest_shape, {"colour": 0, "connectivity": 4, "rank": 0}),
            (GridSelector.independent_cells, {"colour": 0, "connectivity": 4}),
            (GridSelector.outer_border4, {"colour": 0}),
            (GridSelector.inner_border4, {"colour": 0}),
            (GridSelector.grid_border, {"colour": 0}), # Color ignored
            (GridSelector.adjacent4, {"colour": 0, "contacts": 1}),
            (GridSelector.adjacent8, {"colour": 0, "contacts": 1}),
            (self.selector.rectangles, {"colour":0, "height":1, "width":1}),
        ]

        for func, kwargs in test_configs:
            for i, invalid_grid in enumerate(invalid_grids):
                with self.subTest(func_name=func.__name__, input_idx=i):
                    with self.assertRaises(ValueError):
                        func(invalid_grid, **kwargs)

        # Test _check_colour for relevant functions
        funcs_with_color_check = [
            GridSelector.colour, GridSelector.components4, GridSelector.nth_largest_shape,
            GridSelector.independent_cells, GridSelector.outer_border4, GridSelector.adjacent4,
            self.selector.rectangles
        ]
        for func in funcs_with_color_check:
            base_kwargs = {}
            if func == GridSelector.nth_largest_shape: base_kwargs.update({"connectivity":4, "rank":0})
            if func == GridSelector.independent_cells: base_kwargs.update({"connectivity":4})
            if func == GridSelector.adjacent4: base_kwargs.update({"contacts":1})
            if func == self.selector.rectangles: base_kwargs.update({"height":1, "width":1})

            with self.subTest(func_name=func.__name__, check="invalid_color_type"):
                with self.assertRaisesRegex(ValueError, "colour must be a non‑negative integer"):
                    func(self.grid_3x3_standard, colour="abc", **base_kwargs)
            with self.subTest(func_name=func.__name__, check="invalid_color_negative"):
                with self.assertRaisesRegex(ValueError, "colour must be a non‑negative integer"):
                    func(self.grid_3x3_standard, colour=-1, **base_kwargs)

if __name__ == '__main__':
    unittest.main()
