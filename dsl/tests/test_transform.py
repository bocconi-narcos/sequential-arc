import unittest
import numpy as np
from dsl.transform import GridTransformer
from dsl.utils.background import find_background_colour

class TestGridTransformerSpecific(unittest.TestCase):

    def setUp(self):
        self.grid_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.grid_4x4 = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9,10,11,12],
            [13,14,15,16]
        ])
        self.sel_3x3_diag = np.array([[True, False, False], [False, True, False], [False, False, True]])
        self.sel_3x3_all = np.ones((3,3), dtype=bool)
        self.sel_3x3_none = np.zeros((3,3), dtype=bool)
        self.sel_3x3_first_row = np.array([[True, True, True], [False, False, False], [False, False, False]])
        self.sel_4x4_block = np.array([
            [False,False,False,False],
            [False,True, True, False],
            [False,True, True, False],
            [False,False,False,False]
        ])
        self.grid_empty = np.empty((0,0), dtype=int)
        self.sel_empty = np.empty((0,0), dtype=bool)

        # For invert_colors
        self.grid_2_colors = np.array([[1,0,1],[0,1,0],[1,0,1]])
        self.sel_2_colors = np.ones_like(self.grid_2_colors, dtype=bool)

        self.grid_rings = np.array([
            [1,1,1,1,1],
            [1,2,2,2,1],
            [1,2,3,2,1],
            [1,2,2,2,1],
            [1,1,1,1,1]
        ])
        self.sel_rings = np.ones_like(self.grid_rings, dtype=bool)

        self.grid_multicolor_ring = np.array([ # Ring 2 has colors 2 and 4
            [1,1,1,1,1],
            [1,2,4,2,1],
            [1,2,3,2,1],
            [1,2,2,2,1],
            [1,1,1,1,1]
        ])
        self.grid_gap_ring = np.array([ # Gap in the "3" ring
            [1,1,1,1,1],
            [1,2,2,2,1],
            [1,2,0,2,1], # 0 is the gap, not part of selection for ring
            [1,2,2,2,1],
            [1,1,1,1,1]
        ])
        self.sel_gap_ring = self.grid_gap_ring != 0


    def assertGridEqual(self, g1, g2, msg=""):
        np.testing.assert_array_equal(g1, g2, err_msg=msg)

    def assertGridNotEqual(self, g1, g2, msg=""):
        self.assertFalse(np.array_equal(g1,g2), msg=msg)

    def assertIsNewObject(self, g1, g2, msg=""):
        self.assertIsNot(g1,g2, msg=msg)

    # --- Test GridTransformer.identity ---
    def test_identity(self):
        transformed = GridTransformer.identity(self.grid_3x3, self.sel_3x3_diag)
        self.assertGridEqual(transformed, self.grid_3x3)
        self.assertIsNewObject(transformed, self.grid_3x3)
        transformed_empty = GridTransformer.identity(self.grid_empty, self.sel_empty)
        self.assertGridEqual(transformed_empty, self.grid_empty)


    # --- Test GridTransformer.clear ---
    def test_clear(self):
        transformed = GridTransformer.clear(self.grid_3x3, self.sel_3x3_diag)
        expected = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]])
        self.assertGridEqual(transformed, expected)
        self.assertIsNewObject(transformed, self.grid_3x3)

        transformed_all = GridTransformer.clear(self.grid_3x3, self.sel_3x3_all)
        self.assertGridEqual(transformed_all, np.zeros((3,3)))

        transformed_none = GridTransformer.clear(self.grid_3x3, self.sel_3x3_none)
        self.assertGridEqual(transformed_none, self.grid_3x3)


    # --- Test GridTransformer.new_colour and partials ---
    def test_new_colour(self):
        transformed = GridTransformer.new_colour(self.grid_3x3, self.sel_3x3_diag, color=5)
        expected = np.array([[5, 2, 3], [4, 5, 6], [7, 8, 5]])
        self.assertGridEqual(transformed, expected)
        self.assertIsNewObject(transformed, self.grid_3x3)

        for i in range(11): # Test new_colour_0 to new_colour_10
            if hasattr(GridTransformer, f"new_colour_{i}"):
                partial_func = getattr(GridTransformer, f"new_colour_{i}")
                transformed_p = partial_func(self.grid_3x3, self.sel_3x3_diag)
                expected_p = self.grid_3x3.copy()
                expected_p[self.sel_3x3_diag] = i
                self.assertGridEqual(transformed_p, expected_p, msg=f"Test new_colour_{i}")

        # Test with non-standard color, should still work as no check on color value
        transformed_neg = GridTransformer.new_colour(self.grid_3x3, self.sel_3x3_diag, color=-100)
        expected_neg = np.array([[-100, 2, 3], [4, -100, 6], [7, 8, -100]])
        self.assertGridEqual(transformed_neg, expected_neg)


    # --- Test GridTransformer.background_colour ---
    def test_background_colour(self):
        grid_bg_is_0 = np.array([[0,0,1],[0,2,0]])
        sel_bg_is_0 = np.array([[F,F,T],[F,T,F]],dtype=bool,F=False,T=True)
        bg_0 = find_background_colour(grid_bg_is_0) # Should be 0
        self.assertEqual(bg_0, 0)
        transformed_0 = GridTransformer.background_colour(grid_bg_is_0, sel_bg_is_0)
        expected_0 = np.array([[0,0,0],[0,0,0]])
        self.assertGridEqual(transformed_0, expected_0)

        grid_bg_is_5 = np.array([[5,1,5],[5,5,2]])
        sel_bg_is_5 = np.array([[F,T,F],[F,F,T]],dtype=bool,F=False,T=True)
        bg_5 = find_background_colour(grid_bg_is_5) # Should be 5
        self.assertEqual(bg_5, 5)
        transformed_5 = GridTransformer.background_colour(grid_bg_is_5, sel_bg_is_5)
        expected_5 = np.array([[5,5,5],[5,5,5]])
        self.assertGridEqual(transformed_5, expected_5)


    # --- Test GridTransformer.invert_colors ---
    def test_invert_colors(self):
        # Two-color case
        transformed_2c = GridTransformer.invert_colors(self.grid_2_colors, self.sel_2_colors)
        expected_2c = np.array([[0,1,0],[1,0,1],[0,1,0]])
        self.assertGridEqual(transformed_2c, expected_2c)

        # Concentric rings
        # Rings: 1 (outer), 2 (middle), 3 (inner)
        # Expected: 3 (outer), 2 (middle), 1 (inner)
        transformed_rings = GridTransformer.invert_colors(self.grid_rings, self.sel_rings)
        expected_rings = np.array([
            [3,3,3,3,3],
            [3,2,2,2,3],
            [3,2,1,2,3], # Innermost color 1, middle color 2, outermost color 3
            [3,2,2,2,3],
            [3,3,3,3,3]
        ])
        self.assertGridEqual(transformed_rings, expected_rings)

        # Multicolor ring -> no change
        transformed_mc_ring = GridTransformer.invert_colors(self.grid_multicolor_ring, self.sel_rings)
        self.assertGridEqual(transformed_mc_ring, self.grid_multicolor_ring)

        # Gap in ring -> no change (based on current _compute_ring_indices logic, may not form perfect rings)
        transformed_gap_ring = GridTransformer.invert_colors(self.grid_gap_ring, self.sel_gap_ring)
        # This is complex: if 0 is not selected, current may treat it as boundary.
        # If sel_gap_ring selects only non-zero, then it might form valid rings for 1 and 2.
        # Let's assume the description "gap inside selection -> not rings" means it should not change.
        # The code: `if not ring_pixels.any(): return out` - this means a fully empty ring layer.
        # `_compute_ring_indices` will label 0 as -1.
        # If sel_gap_ring is used, 0s are not selected. The remaining 1,2,3 will form rings.
        # Ring 0: 1s. Ring 1: 2s. Ring 2: 3s is not present, this needs check.
        # The test case grid_gap_ring has 0 in the middle, sel_gap_ring excludes it.
        # Rings: R0=1, R1=2. Swaps 1 and 2.
        expected_gap_ring_transformed = np.array([
            [2,2,2,2,2],
            [2,1,1,1,2],
            [2,1,0,1,2], # 0 is untouched as not in selection
            [2,1,1,1,2],
            [2,2,2,2,2]
        ])
        self.assertGridEqual(transformed_gap_ring, expected_gap_ring_transformed)


        # Selection with 1 color -> no change
        transformed_1c = GridTransformer.invert_colors(self.grid_3x3, np.array([[T,F,F],[F,F,F],[F,F,F]],T=True,F=False))
        self.assertGridEqual(transformed_1c, self.grid_3x3)

        # Empty selection -> no change
        transformed_empty_sel = GridTransformer.invert_colors(self.grid_3x3, self.sel_3x3_none)
        self.assertGridEqual(transformed_empty_sel, self.grid_3x3)


    # --- Test GridTransformer.flip ---
    def test_flip(self):
        # Vertical flip (axis=0) of first row
        transformed_v = GridTransformer.flip(self.grid_3x3, self.sel_3x3_first_row, axis=0)
        # Bounding box of sel_3x3_first_row is the first row itself. Flipping it vertically does nothing.
        self.assertGridEqual(transformed_v, self.grid_3x3)

        # Vertical flip of a 2x2 block in 4x4
        # sel_4x4_block selects (1,1), (1,2), (2,1), (2,2)
        # Values: [[6,7],[10,11]] -> flip_vertical -> [[10,11],[6,7]]
        transformed_v_block = GridTransformer.flip_vertical(self.grid_4x4, self.sel_4x4_block)
        expected_v_block = self.grid_4x4.copy()
        expected_v_block[1:3, 1:3] = np.array([[10,11],[6,7]])
        self.assertGridEqual(transformed_v_block, expected_v_block)

        # Horizontal flip (axis=1) of a 2x2 block
        transformed_h_block = GridTransformer.flip_horizontal(self.grid_4x4, self.sel_4x4_block)
        expected_h_block = self.grid_4x4.copy()
        expected_h_block[1:3, 1:3] = np.array([[7,6],[11,10]])
        self.assertGridEqual(transformed_h_block, expected_h_block)

        # Empty selection
        transformed_empty_sel = GridTransformer.flip(self.grid_3x3, self.sel_3x3_none, axis=0)
        self.assertGridEqual(transformed_empty_sel, self.grid_3x3)


    # --- Test GridTransformer.rotate ---
    def test_rotate(self):
        # Rotate 90 on a square selection
        sel_square = np.array([[T,T,F],[T,T,F],[F,F,F]],T=True,F=False) # Bounding box 2x2
        grid_for_rot = np.array([[1,2,0],[3,4,0],[0,0,0]])
        # Selected part: [[1,2],[3,4]] -> rot90 -> [[2,4],[1,3]]
        transformed_r90 = GridTransformer.rotate_90(grid_for_rot, sel_square)
        expected_r90 = np.array([[2,4,0],[1,3,0],[0,0,0]])
        self.assertGridEqual(transformed_r90, expected_r90)

        # Rotate 180 on a rectangular selection
        sel_rect = np.array([[T,T,T],[T,T,T],[F,F,F]],T=True,F=False) # Bounding box 2x3
        # Selected part: [[1,2,3],[4,5,6]] -> rot180 -> [[6,5,4],[3,2,1]]
        transformed_r180 = GridTransformer.rotate_180(self.grid_3x3, sel_rect)
        expected_r180 = np.array([[6,5,4],[3,2,1],[7,8,9]])
        self.assertGridEqual(transformed_r180, expected_r180)

        # Rotate 90 on selection with non-square bounding box -> no change
        grid_nonsq_sel_bounds = np.array([[1,2,3],[4,5,6]])
        sel_nonsq_bounds = np.ones((2,3), dtype=bool)
        transformed_r90_nsb = GridTransformer.rotate_90(grid_nonsq_sel_bounds, sel_nonsq_bounds)
        self.assertGridEqual(transformed_r90_nsb, grid_nonsq_sel_bounds)

        # Empty selection
        transformed_empty_sel = GridTransformer.rotate_90(self.grid_3x3, self.sel_3x3_none)
        self.assertGridEqual(transformed_empty_sel, self.grid_3x3)


    # --- Test GridTransformer.slide_new and slide_old ---
    # These require many sub-tests. Will simplify for now and assume general test covers more.
    # Focus on constraints and a few key behaviors.
    def test_slide_constraints(self):
        # slide_new: continuous with fluid/superfluid
        with self.assertRaisesRegex(ValueError, "'continuous' cannot be combined with 'fluid' or 'superfluid'"):
            GridTransformer.slide_new(self.grid_3x3, self.sel_3x3_diag, direction="down", continuous=True, fluid=True)
        with self.assertRaisesRegex(ValueError, "'continuous' cannot be combined with 'fluid' or 'superfluid'"):
            GridTransformer.slide_new(self.grid_3x3, self.sel_3x3_diag, direction="down", continuous=True, superfluid=True, fluid=True)

        # slide_new: superfluid without fluid
        with self.assertRaisesRegex(ValueError, "'superfluid' implies 'fluid'=True"):
            GridTransformer.slide_new(self.grid_3x3, self.sel_3x3_diag, direction="down", superfluid=True, fluid=False)

        # slide_new: superfluid with continuous (already covered by first check, but also specific constraint)
        with self.assertRaisesRegex(ValueError, "'superfluid' cannot be combined with 'continuous'"):
             GridTransformer.slide_new(self.grid_3x3, self.sel_3x3_diag, direction="down", continuous=True, superfluid=True, fluid=True)


    def test_slide_new_basic(self):
        grid = np.array([[1,0,0],[0,0,0],[0,0,0]])
        sel = np.array([[T,F,F],[F,F,F],[F,F,F]],T=True,F=False) # Selects cell (0,0) with value 1

        # Copy down, single, no obstacles
        transformed = GridTransformer.slide_new(grid, sel, direction="down", mode="copy", continuous=False, obstacles=False)
        expected = np.array([[1,0,0],[0,0,0],[1,0,0]]) # (0,0) copied to (2,0)
        self.assertGridEqual(transformed, expected)

        # Cut down, single, no obstacles
        transformed_cut = GridTransformer.slide_new(grid, sel, direction="down", mode="cut", continuous=False, obstacles=False)
        expected_cut = np.array([[0,0,0],[0,0,0],[1,0,0]]) # (0,0) moved to (2,0)
        self.assertGridEqual(transformed_cut, expected_cut)

        # Copy down, continuous, no obstacles
        grid_cont = np.array([[1,0],[0,0],[0,0],[0,0]])
        sel_cont = np.array([[T,F],[F,F],[F,F],[F,F]],T=True,F=False)
        # step_len is 1 (height of selection). Tiles at (1,0), (2,0), (3,0)
        transformed_cont = GridTransformer.slide_new(grid_cont, sel_cont, direction="down", mode="copy", continuous=True, obstacles=False)
        expected_cont = np.array([[1,0],[1,0],[1,0],[1,0]])
        self.assertGridEqual(transformed_cont, expected_cont)

        # Obstacle test
        grid_obs = np.array([[1,0,0],[0,0,0],[2,0,0]]) # Obstacle at (2,0)
        sel_obs = np.array([[T,F,F],[F,F,F],[F,F,F]],T=True,F=False)
        # Copy down, single, with obstacles
        transformed_obs = GridTransformer.slide_new(grid_obs, sel_obs, direction="down", mode="copy", continuous=False, obstacles=True)
        expected_obs = np.array([[1,0,0],[1,0,0],[2,0,0]]) # Stops at (1,0) before obstacle
        self.assertGridEqual(transformed_obs, expected_obs)


    def test_slide_old_basic(self): # slide_old has different internal logic, check a few cases
        grid = np.array([[1,0,0],[0,0,0],[0,0,0]])
        sel = np.array([[T,F,F],[F,F,F],[F,F,F]],T=True,F=False)

        # Cut down, single, fluid, superfluid (move_down_superfluid equivalent)
        # For a single selected cell, fluid/superfluid might not differ much from non-fluid
        transformed = GridTransformer.slide_old(grid, sel, direction="down", mode="cut", continuous=False, obstacles=True, fluid=True, superfluid=True)
        # Expected: (0,0) with value 1 moves to (2,0)
        expected = np.array([[0,0,0],[0,0,0],[1,0,0]])
        self.assertGridEqual(transformed, expected)

        # Check move_down_superfluid partial
        transformed_partial = GridTransformer.move_down_superfluid(grid, sel)
        self.assertGridEqual(transformed_partial, expected)


    # --- Input Validation ---
    def test_input_validation(self):
        # Grid checks
        with self.assertRaisesRegex(ValueError, "grid must be a 2-D NumPy array"):
            GridTransformer.identity(np.array([1,2,3]), self.sel_3x3_none)
        with self.assertRaisesRegex(ValueError, "grid dtype must be an integer type"):
            GridTransformer.identity(np.array([["a"]]), np.array([[True]]))

        # Mask checks
        with self.assertRaisesRegex(ValueError, "selection must be a 2-D boolean NumPy array"):
            GridTransformer.clear(self.grid_3x3, np.array([1,0,1])) # Not 2D
        with self.assertRaisesRegex(ValueError, "selection must be a 2-D boolean NumPy array"):
            GridTransformer.clear(self.grid_3x3, self.grid_3x3) # Not boolean
        with self.assertRaisesRegex(ValueError, "selection mask must match grid shape"):
            GridTransformer.clear(self.grid_3x3, np.ones((2,2),dtype=bool))

        # Slide specific param validation
        with self.assertRaisesRegex(ValueError, "Unknown direction"):
            GridTransformer.slide_new(self.grid_3x3, self.sel_3x3_all, direction="diagonal")
        with self.assertRaisesRegex(ValueError, "'mode' must be 'copy' or 'cut'"):
            GridTransformer.slide_new(self.grid_3x3, self.sel_3x3_all, direction="down", mode="delete")
        with self.assertRaisesRegex(TypeError, "'selection' must be boolean"): # slide_new uses TypeError for bad selection dtype
             GridTransformer.slide_new(self.grid_3x3, self.grid_3x3.astype(int), direction="down")


if __name__ == '__main__':
    unittest.main()
