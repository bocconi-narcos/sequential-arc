"""
Grid-transformation utilities for ARC-style tasks and Gymnasium-compatible
research environments.

Contract
────────
    transform(grid: np.ndarray,
              selection: np.ndarray[bool],  # 2-D mask, *never* promoted
              **kwargs) -> np.ndarray

Returned value is always a *new* 2-D integer grid (no in-place mutation).

"""
from __future__ import annotations

from functools import partial
from typing import Mapping, Tuple

import numpy as np
from scipy.ndimage import binary_fill_holes, label   # kept for future use

from dsl.utils.background import find_background_colour

# ───────────────────────────────────────────────────────────────────── #
# Validation helpers
# ───────────────────────────────────────────────────────────────────── #
def _check_grid(grid: np.ndarray) -> None:
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        raise ValueError("grid must be a 2-D NumPy array")
    if not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("grid dtype must be an integer type")


def _check_mask(mask: np.ndarray, grid_shape: Tuple[int, int]) -> None:
    if not (isinstance(mask, np.ndarray) and mask.dtype == bool and mask.ndim == 2):
        raise ValueError("selection must be a 2-D boolean NumPy array")
    if mask.shape != grid_shape:
        raise ValueError("selection mask must match grid shape")

# ───────────────────────────────────────────────────────────────────── #
# Geometry helpers
# ───────────────────────────────────────────────────────────────────── #
def _copy(grid: np.ndarray) -> np.ndarray:
    return np.array(grid, copy=True)


def _bounding_rectangle(mask: np.ndarray) -> Tuple[slice, slice]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return slice(0, 0), slice(0, 0)
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def _bounding_square(mask: np.ndarray) -> Tuple[slice, slice]:
    rs, cs = _bounding_rectangle(mask)
    # if not a square return false mask
    if rs.stop - rs.start != cs.stop - cs.start:
        return slice(0, 0), slice(0, 0)
    return rs, cs

# ───────────────────────────────────────────────────────────────────── #
# Primitive copy / cut-and-paste (2-D)
# ───────────────────────────────────────────────────────────────────── #
def _paste(
    grid: np.ndarray,
    mask: np.ndarray,
    shift_y: int,
    shift_x: int,
    *,
    cut: bool = False,
) -> np.ndarray:
    """Copy or cut-and-paste *mask* by (shift_y, shift_x)."""
    out = _copy(grid)
    rows, cols = grid.shape

    ys, xs = np.where(mask)
    new_y = ys + shift_y
    new_x = xs + shift_x

    #print('new x: ', new_x)
    #print('new_y: ', new_y)

    # Boundary check
    if (
        new_y.min() < 0
        or new_y.max() >= rows
        or new_x.min() < 0
        or new_x.max() >= cols
    ):
        return out  # invalid → no-op

    if cut:
        out[ys, xs] = 0

    #collision = (out[new_y, new_x] != 0) & (~mask[new_y, new_x])
    #if collision.any():
    #    return _copy(grid)  # invalid → no-op

    out[new_y, new_x] = grid[ys, xs]
    return out

# ───────────────────────────────────────────────────────────────────── #
# Transformer library
# ───────────────────────────────────────────────────────────────────── #
class GridTransformer:  # noqa: WPS110
    """Stateless collection of grid transformations."""

    __slots__ = ()

    # 0️⃣ Identity / Clear
    # -------------------
    @staticmethod
    def identity(grid: np.ndarray, *_a, **_k) -> np.ndarray:
        _check_grid(grid)
        return _copy(grid)

    @staticmethod
    def clear(grid: np.ndarray, selection: np.ndarray, **__) -> np.ndarray:
        _check_grid(grid)
        _check_mask(selection, grid.shape)
        out = _copy(grid)
        out[selection] = 0
        return out
    
    @staticmethod
    def new_colour(grid: np.ndarray, selection: np.ndarray, color: int, **__) -> np.ndarray:
        _check_grid(grid)
        _check_mask(selection, grid.shape)
        out = _copy(grid)
        out[selection] = color
        return out

    new_colour_0, new_colour_0.__name__ = partial(new_colour, color=0), "new_colour_0"
    new_colour_1, new_colour_1.__name__ = partial(new_colour, color=1), "new_colour_1"
    new_colour_2, new_colour_2.__name__ = partial(new_colour, color=2), "new_colour_2"
    new_colour_3, new_colour_3.__name__ = partial(new_colour, color=3), "new_colour_3"
    new_colour_4, new_colour_4.__name__ = partial(new_colour, color=4), "new_colour_4"
    new_colour_5, new_colour_5.__name__ = partial(new_colour, color=5), "new_colour_5"
    new_colour_6, new_colour_6.__name__ = partial(new_colour, color=6), "new_colour_6"
    new_colour_7, new_colour_7.__name__ = partial(new_colour, color=7), "new_colour_7"
    new_colour_8, new_colour_8.__name__ = partial(new_colour, color=8), "new_colour_8"
    new_colour_9, new_colour_9.__name__ = partial(new_colour, color=9), "new_colour_9"
    new_colour_10, new_colour_10.__name__ = partial(new_colour, color=10), "new_colour_10"

    @staticmethod
    def background_colour(grid: np.ndarray, selection: np.ndarray, **__) -> np.ndarray:
        _check_grid(grid)
        _check_mask(selection, grid.shape)
        
        _back_ground_colour = find_background_colour(grid)
        out = _copy(grid)
        out[selection] = _back_ground_colour
        return out
    
    background_colour, background_colour.__name__ = partial(background_colour), "background_colour"

    @staticmethod
    def invert_colors(grid: np.ndarray, selection: np.ndarray, **__) -> np.ndarray:
        """
        If exactly two distinct colors appear in *selection*, swap them (as before).

        If more than two colors appear, check whether the selected pixels form
        *concentric rings*:

        *  A **ring** is the set of pixels whose graph-distance from the exterior of
        the selection is the same.  Rings can be any thickness ≥ 1, and several
        non-adjacent rings may share the same colour.
        *  All pixels that belong to the same ring must have a **single unique
        colour**.  If any ring is multicoloured the selection is **not** a valid
        concentric-ring pattern and the grid is left unchanged.

        When the rings condition holds, colours are mirrored:

        *  Outermost ring ↔ innermost ring
        *  Second-outer ↔ second-inner
        *  …and so on.  
        With an odd number of rings the middle one keeps its colour.

        Otherwise the grid is returned unchanged.
        """
        _check_grid(grid)
        _check_mask(selection, grid.shape)

        out = _copy(grid)
        # ------------------------------------------------------------------ #
        # 1. Fast path: exactly two colours → same logic as the original code
        # ------------------------------------------------------------------ #
        colours = np.unique(grid[selection])
        if colours.size == 2:                       # unchanged behaviour
            c1, c2 = colours
            sel = selection
            out[sel & (grid == c1)] = c2
            out[sel & (grid == c2)] = c1
            return out

        # ------------------------------------------------------------------ #
        # 2. Concentric-rings logic for 3 + colours
        # ------------------------------------------------------------------ #
        # Utility: label each selected pixel with its *ring index* (0 = outermost)
        def _compute_ring_indices(mask: np.ndarray) -> np.ndarray:
            """
            Label each pixel in *mask* with its ring depth.
                -1  : not in mask
                0… : outermost, …, innermost ring
            A “ring” is the set of pixels that touch the exterior after the previous
            rings have been removed (4-neighbour connectivity).
            """
            rings = np.full(mask.shape, -1, dtype=int)
            if not mask.any():                       # empty selection – nothing to do
                return rings

            current = mask.copy()                    # pixels that still need a label
            depth   = 0

            while current.any():
                # ---- identify boundary pixels (touch False or the grid border) ----
                padded  = np.pad(current, 1, constant_values=False)
                up      = ~padded[:-2, 1:-1]
                down    = ~padded[2:,  1:-1]
                left    = ~padded[1:-1, :-2]
                right   = ~padded[1:-1, 2:]
                boundary = current & (up | down | left | right)

                # If *everything* is interior (e.g. mask fills the whole grid),
                # treat the remainder as the last ring and stop.
                if not boundary.any():
                    rings[current] = depth
                    break
                

                rings[boundary] = depth
                current &= ~boundary                 # peel off the layer
                depth   += 1

            return rings

        ring_idx = _compute_ring_indices(selection)
        if (ring_idx == -1).all():          # selection empty – nothing to do
            return out

        n_rings = ring_idx.max() + 1

        # Verify that every ring is monochrome
        ring_colours = []
        for d in range(n_rings):
            ring_pixels = (ring_idx == d)
            if not ring_pixels.any():         # gap inside selection → not rings
                return out
            uniq = np.unique(grid[ring_pixels])
            if uniq.size != 1:                # multicoloured ring
                return out
            ring_colours.append(uniq[0])

        # ------------------------------------------------------------------ #
        # 3. Mirror colours: outermost ↔ innermost, etc.
        # ------------------------------------------------------------------ #
        for d in range(n_rings // 2):
            inner = n_rings - 1 - d
            c_outer, c_inner = ring_colours[d], ring_colours[inner]

            outer_mask = (selection & (ring_idx == d) & (grid == c_outer))
            inner_mask = (selection & (ring_idx == inner) & (grid == c_inner))

            out[outer_mask] = c_inner
            out[inner_mask] = c_outer

        return out

    # wrap & rename so signature hides the internal **__ argument
    invert_colors, invert_colors.__name__ = partial(invert_colors), "invert_colors"

    # 1️⃣ Flips
    # ---------
    @staticmethod
    def flip(grid: np.ndarray, selection: np.ndarray, *, axis: int, **__) -> np.ndarray:
        """
        Mirror *selection* around vertical (axis=0) or horizontal (axis=1) axis.
        """
        _check_grid(grid)
        _check_mask(selection, grid.shape)
        out = _copy(grid)

        rs, cs = _bounding_rectangle(selection)
        if rs.stop == rs.start:  # empty mask
            return out
        sub = out[rs, cs]
        sel_sub = selection[rs, cs]
        flipped = np.flip(sub, axis=axis)
        sub[sel_sub] = flipped[sel_sub]
        out[rs, cs] = sub
        return out

    flip_vertical, flip_vertical.__name__ = partial(flip, axis=0), "flip_vertical"
    flip_horizontal, flip_horizontal.__name__ = partial(flip, axis=1), "flip_horizontal"

    # 2️⃣ Rotations (square-only)
    # ---------------------------
    @staticmethod
    def rotate(
        grid: np.ndarray,
        selection: np.ndarray,
        *,
        k: int = 1,
        **__,
        ) -> np.ndarray:
        """
        Rotate *selection* by k·90° CCW.

        Uses the bounding square if rotation is not 180°.
        Uses the bounding rectangle if rotation is exactly 180°.
        """
        _check_grid(grid)
        _check_mask(selection, grid.shape)
        out = _copy(grid)

        if k % 4 == 2:
            # 180° rotation: use bounding rectangle
            rect_rs, rect_cs = _bounding_rectangle(selection)
            sub = out[rect_rs, rect_cs]
            sel_sub = selection[rect_rs, rect_cs]
        else:
            # Other rotations: use bounding square
            square_rs, square_cs = _bounding_square(selection)
            sub = out[square_rs, square_cs]
            sel_sub = selection[square_rs, square_cs]
            if sub.shape[0] != sub.shape[1] or not sel_sub.any():
                return out

        rotated = np.rot90(sub, k)
        sub = rotated
        
        if k % 4 == 2:
            out[rect_rs, rect_cs] = sub
        else:
            out[square_rs, square_cs] = sub

        return out

    rotate_90, rotate_90.__name__ = partial(rotate, k=1), "rotate_90"
    rotate_180, rotate_180.__name__ = partial(rotate, k=2), "rotate_180"
    rotate_270, rotate_270.__name__ = partial(rotate, k=3), "rotate_270"
   
    @staticmethod
    def slide_new(
        grid: np.ndarray,
        selection: np.ndarray,
        *,
        direction: str,            # "up" | "right" | "down" | "left"
        mode: str = "copy",        # "copy" | "cut"
        continuous: bool = True,   # paste every step vs. single placement
        obstacles: bool = True,    # treat filled cells as blockers
        fluid: bool = False,       # treat each column/row independently
        superfluid: bool = False   # stack fragmented strips into a single block
    ) -> np.ndarray:
        """Translate **selection** through **grid** along **direction**.

        Parameters
        ----------
        grid : ndarray (2‑D, int)
            The canvas to draw on. ``0`` is assumed to be *empty*; any other
            value is considered *occupied* and is re‑drawn verbatim unless
            *mode='cut'* clears it.
        selection : ndarray (bool, same shape as **grid**)
            Boolean mask indicating which cells are candidates for copying/
            moving.
        direction : {"up", "right", "down", "left"}
            The direction of the translation.
        mode : {"copy", "cut"}, default *"copy"*
            If *"cut"*, the original **selection** cells are cleared **once**
            after the new positions have been placed.
        continuous : bool, default *True*
            If *True*, the object is *tiled* every *step‑length* (its own
            width/height) until nothing more fits inside the grid.  When
            *False*, the object is moved **once**, as far as possible without
            clipping or hitting an obstacle.
        obstacles : bool, default *True*
            When *True*, already occupied cells stop the movement.  With
            *continuous=True* we simply *skip* blocked destination cells; with
            *continuous=False* we stop at the last fully unobstructed
            position.
        fluid : bool, default *False*
            Treats *each column* (for vertical moves) or *each row* (for
            horizontal moves) as an independent strip and slides them
            separately.  Behaviour is otherwise identical to the non‑fluid
            case.
        superfluid : bool, default *False*
            A stricter variant of *fluid* that **stacks** every *fragment* of a
            strip into a single contiguous block before placing it.  For
            example, if a vertical move encounters the pattern ``##__###`` in
            one column (``#`` = selected, ``_`` = unselected), the five hash
            cells are first collapsed into the shape ``#####`` and then slid
            en bloc until they settle – mimicking a *gravity* effect.  Only
            the destination strip is compacted; the original layout is left
            intact (unless *mode='cut'* clears it at the end).

        Constraints
        -----------
        * ``continuous`` *cannot* be combined with ``fluid`` or ``superfluid``.
        * ``superfluid`` implies ``fluid`` – enabling one without the other
          raises a :class:`ValueError`.

        Returns
        -------
        ndarray
            A *copy* of **grid** with the translation applied.
        """
        # ───────────────────── validation ─────────────────────
        if grid.ndim != 2:
            raise ValueError("'grid' must be 2‑D")
        if selection.shape != grid.shape:
            raise ValueError("'selection' mask must match 'grid' shape")
        if selection.dtype != bool:
            raise TypeError("'selection' must be boolean")

        direction = direction.lower()
        mode      = mode.lower()
        if direction not in {"up", "right", "down", "left"}:
            raise ValueError(f"Unknown direction {direction!r}")
        if mode not in {"copy", "cut"}:
            raise ValueError("'mode' must be 'copy' or 'cut'")

        # ───────────── new cross‑parameter constraints ─────────────
        if continuous and (fluid or superfluid):
            raise ValueError("'continuous' cannot be combined with 'fluid' or 'superfluid'")
        if superfluid and not fluid:
            raise ValueError("'superfluid' implies 'fluid'=True")
        if superfluid and continuous:
            raise ValueError("'superfluid' cannot be combined with 'continuous'")

        # ───────────────────── helpers ─────────────────────
        # Direction vector
        drow, dcol = {
            "up":    (-1,  0),
            "right": ( 0,  1),
            "down":  ( 1,  0),
            "left":  ( 0, -1),
        }[direction]

        n_rows, n_cols = grid.shape

        def in_bounds(rr: int, cc: int) -> bool:
            return 0 <= rr < n_rows and 0 <= cc < n_cols

        # ───────────────────── inner core ─────────────────────
        def _slide_core(_grid: np.ndarray, _sel_mask: np.ndarray) -> None:
            """In‑place application of the sliding rules for *one* contiguous mask.

            Always works in *copy* mode to avoid clearing the original cells of
            other strips.  *CUT* handling is done once at the very end by the
            outer function.
            """
            # Coordinates of *this* sub‑selection
            sel_rows, sel_cols = np.nonzero(_sel_mask)
            if sel_rows.size == 0:
                return  # nothing to do
            sel_vals = _grid_copy_source[sel_rows, sel_cols]

            # Helper lambdas working *vectorised* on arrays
            def blocked(rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
                if not obstacles:
                    return np.zeros(rr.shape, bool)
                return _grid[rr, cc] != 0

            # Step length (width or height of the current sub‑mask)
            step_len = (
                (np.ptp(sel_cols) + 1) if direction in {"left", "right"}
                else (np.ptp(sel_rows) + 1)
            )
            #print('sel_mask: ', _sel_mask)
            #print('sel_rows: ', sel_rows)
            #print('sel_cols: ', sel_cols)
            #print('sel_vals: ', sel_vals)
            #print('step_len: ', step_len)

            if continuous:  # ───────── tile repeatedly ─────────
                step_idx = 1
                while True:
                    r_off = step_idx * step_len * drow
                    c_off = step_idx * step_len * dcol
                    dest_rows = sel_rows + r_off
                    dest_cols = sel_cols + c_off

                    inside = (
                        (dest_rows >= 0) & (dest_rows < n_rows) &
                        (dest_cols >= 0) & (dest_cols < n_cols)
                    )
                    if not inside.any():
                        break  # nothing left inside grid

                    blk = blocked(dest_rows[inside], dest_cols[inside])
                    for i_in, i_sel in enumerate(np.flatnonzero(inside)):
                        if blk[i_in]:
                            continue
                        r = int(dest_rows[i_sel]); c = int(dest_cols[i_sel])
                        if _visited_global[r, c]:
                            continue
                        _grid[r, c] = int(sel_vals[i_sel])
                        _visited_global[r, c] = True

                    if obstacles and blk.all():
                        break  # whole tile blocked – stop early
                    step_idx += 1

            else:          # ─────── single placement ────────
                offset = 0
                while True:
                    trial_off = offset + 1
                    r_off = trial_off * drow
                    c_off = trial_off * dcol
                    dest_rows = sel_rows + r_off
                    dest_cols = sel_cols + c_off

                    if not (
                        (dest_rows >= 0) & (dest_rows < n_rows) &
                        (dest_cols >= 0) & (dest_cols < n_cols)
                    ).all():
                        break  # would clip
                    if blocked(dest_rows, dest_cols).any():
                        break  # obstacle ahead
                    offset = trial_off

                if offset > 0:
                    r_off = offset * drow
                    c_off = offset * dcol
                    dest_rows = sel_rows + r_off
                    dest_cols = sel_cols + c_off
                    _grid[dest_rows, dest_cols] = sel_vals
                    _visited_global[dest_rows, dest_cols] = True

        # ───────────── superfluid helpers ─────────────
        def _stack_and_slide_vertical(_grid: np.ndarray, col: int) -> None:
            """Compact *all* selected cells in **col** and let them *fall/rise*."""
            sel_rows = np.flatnonzero(selection[:, col])
            if sel_rows.size == 0:
                return
            # Order of processing: bottom‑to‑top for 'down', top‑to‑bottom for 'up'
            proc_order = sel_rows[::-1] if direction == "down" else sel_rows
            for r in proc_order:
                val = _grid_copy_source[r, col]
                cur_r = r
                while True:
                    nxt_r = cur_r + drow
                    if not in_bounds(nxt_r, col):
                        break  # hit grid border
                    # Blocking rule: visited cells *always* block; other cells
                    # block only if 'obstacles' is True.
                    if _visited_global[nxt_r, col]:
                        break
                    if obstacles and _grid[nxt_r, col] != 0 and not _visited_global[nxt_r, col]:
                        break
                    cur_r = nxt_r
                # Place the copy if it moved at all or if destination not yet occupied
                if not _visited_global[cur_r, col]:
                    _grid[cur_r, col] = int(val)
                    _visited_global[cur_r, col] = True

        def _stack_and_slide_horizontal(_grid: np.ndarray, row: int) -> None:
            """Compact *all* selected cells in **row** and let them slide left/right."""
            sel_cols = np.flatnonzero(selection[row])
            if sel_cols.size == 0:
                return
            proc_order = sel_cols[::-1] if direction == "right" else sel_cols
            for c in proc_order:
                val = _grid_copy_source[row, c]
                cur_c = c
                while True:
                    nxt_c = cur_c + dcol
                    if not in_bounds(row, nxt_c):
                        break
                    if _visited_global[row, nxt_c]:
                        break
                    if obstacles and _grid[row, nxt_c] != 0 and not _visited_global[row, nxt_c]:
                        break
                    cur_c = nxt_c
                if not _visited_global[row, cur_c]:
                    _grid[row, cur_c] = int(val)
                    _visited_global[row, cur_c] = True

        # ──────────────────── main body ───────────────────────
        _grid_copy_source = grid.copy()  # reference for original colours
        out = grid.copy()
        _visited_global = np.zeros_like(selection, bool)

        if not fluid:                    # ────── original behaviour ──────
            _slide_core(out, selection)

        elif superfluid:                 # ────── superfluid stacking ─────
            if direction in {"up", "down"}:   # independent *columns*
                for col_idx in np.flatnonzero(selection.any(axis=0)):
                    _stack_and_slide_vertical(out, col_idx)
            else:                                # horizontal ⇒ independent rows
                for row_idx in np.flatnonzero(selection.any(axis=1)):
                    _stack_and_slide_horizontal(out, row_idx)

        else:                           # ────── regular fluid ──────
            if direction in {"up", "down"}:   # treat each *column* independently
                for col_idx in np.flatnonzero(selection.any(axis=0)):
                    sub_mask = selection & (np.arange(grid.shape[1]) == col_idx)
                    _slide_core(out, sub_mask)
            else:                               # left / right ⇒ independent rows
                for row_idx in np.flatnonzero(selection.any(axis=1)):
                    sub_mask = selection & (np.arange(grid.shape[0])[:, None] == row_idx)
                    _slide_core(out, sub_mask)

        # ───────────── CUT mode (done once!) ─────────────
        if mode == "cut":
            out[selection] = 0

        return out

    """
    # Cut and paste not fluid
    move_single_up_block    = partial(slide, direction="up",    mode="cut",  continuous=False, fluid=False,)
    move_single_down_block  = partial(slide, direction="down",  mode="cut",  continuous=False, fluid=False)
    move_single_left_block  = partial(slide, direction="left",  mode="cut",  continuous=False, fluid=False)
    move_single_right_block = partial(slide, direction="right", mode="cut",  continuous=False, fluid=False)
    
    move_single_up_block.__name__    = "move_single_up_block"
    move_single_down_block.__name__  = "move_single_down_block"
    move_single_left_block.__name__  = "move_single_left_block"
    move_single_right_block.__name__ = "move_single_right_block"

    # Cut and paste fluid
    move_single_up_fluid    = partial(slide, direction="up",    mode="cut",  continuous=False, fluid=True)
    move_single_down_fluid  = partial(slide, direction="down",  mode="cut",  continuous=False, fluid=True)
    move_single_left_fluid  = partial(slide, direction="left",  mode="cut",  continuous=False, fluid=True)
    move_single_right_fluid = partial(slide, direction="right", mode="cut",  continuous=False, fluid=True)
    move_single_up_fluid.__name__    = "move_single_up_fluid"
    move_single_down_fluid.__name__  = "move_single_down_fluid"
    move_single_left_fluid.__name__  = "move_single_left_fluid"
    move_single_right_fluid.__name__ = "move_single_right_fluid"

    # Copy and paste not fluid
    clone_single_up_block    = partial(slide, direction="up",    mode="copy", continuous=False, fluid=False)
    clone_single_down_block  = partial(slide, direction="down",  mode="copy", continuous=False, fluid=False)
    clone_single_left_block  = partial(slide, direction="left",  mode="copy", continuous=False, fluid=False)
    clone_single_right_block = partial(slide, direction="right", mode="copy", continuous=False, fluid=False)
    clone_single_up_block.__name__    = "clone_single_up_block"
    clone_single_down_block.__name__  = "clone_single_down_block"
    clone_single_left_block.__name__  = "clone_single_left_block"
    clone_single_right_block.__name__ = "clone_single_right_block"
    clone_single_up_block.__name__    = "clone_single_up_block"

    # Copy and paste fluid
    clone_single_up_fluid    = partial(slide, direction="up",    mode="copy", continuous=False, fluid=True)
    clone_single_down_fluid  = partial(slide, direction="down",  mode="copy", continuous=False, fluid=True)
    clone_single_left_fluid  = partial(slide, direction="left",  mode="copy", continuous=False, fluid=True)
    clone_single_right_fluid = partial(slide, direction="right", mode="copy", continuous=False, fluid=True)
    clone_single_up_fluid.__name__    = "clone_single_up_fluid"
    clone_single_down_fluid.__name__  = "clone_single_down_fluid"
    clone_single_left_fluid.__name__  = "clone_single_left_fluid"
    clone_single_right_fluid.__name__ = "clone_single_right_fluid"
    
    # Copy and paste continuous not fluid
    clone_multi_up_block    = partial(slide, direction="up",    mode="copy", continuous=True, fluid=False)
    clone_multi_down_block  = partial(slide, direction="down",  mode="copy", continuous=True, fluid=False)
    clone_multi_left_block  = partial(slide, direction="left",  mode="copy", continuous=True, fluid=False)
    clone_multi_right_block = partial(slide, direction="right", mode="copy", continuous=True, fluid=False)
    clone_multi_up_block.__name__    = "clone_multi_up_block"
    clone_multi_down_block.__name__  = "clone_multi_down_block"
    clone_multi_left_block.__name__  = "clone_multi_left_block"
    clone_multi_right_block.__name__ = "clone_multi_right_block"

    # Copy and paste continuous fluid
    clone_multi_up_fluid    = partial(slide, direction="up",    mode="copy", continuous=True, fluid=True)
    clone_multi_down_fluid  = partial(slide, direction="down",  mode="copy", continuous=True, fluid=True)
    clone_multi_left_fluid  = partial(slide, direction="left",  mode="copy", continuous=True, fluid=True)
    clone_multi_right_fluid = partial(slide, direction="right", mode="copy", continuous=True, fluid=True)
    clone_multi_up_fluid.__name__    = "clone_multi_up_fluid"
    clone_multi_down_fluid.__name__  = "clone_multi_down_fluid"
    clone_multi_left_fluid.__name__  = "clone_multi_left_fluid"
    clone_multi_right_fluid.__name__ = "clone_multi_right_fluid"
    """


    @staticmethod
    def slide_old(
        grid: np.ndarray,
        selection: np.ndarray,
        *,
        direction: str,            # "up" | "right" | "down" | "left"
        mode: str = "copy",        # "copy" | "cut"
        continuous: bool = True,   # paste every step vs. single placement
        obstacles: bool = True,    # treat filled cells as blockers
        fluid: bool = False,        # treat each column/row independently
        superfluid: bool = True   # stack fragmented strips into a single block
    ) -> np.ndarray:
        """Translate *selection* through *grid* along *direction*.

        Parameters added/changed
        ------------------------
        fluid : bool, default **False**
            * **False** – behaviour identical to previous version.
            * **True** – the mask is **decomposed** into independent strips:
                columns when moving *up/down* and rows when moving *left/right*.
                Each strip is slid separately using the same rules for
                *continuous*, *obstacles*, etc.

        Summary of existing behaviour
        -----------------------------
        * *continuous=True*  ⇒ tile the object every *step‑length* (its own
            width/height) until no part remains in bounds – the last tile may be
            clipped.
        * *continuous=False* ⇒ slide once, as far as possible **without
            clipping**, then stop.
        * *obstacles=True*   ⇒ previously filled cells block the move.  With
            *continuous=True* we simply skip the blocked destination cells; with
            *continuous=False* we stop at the last fully unobstructed position.
        """
        # ───────────────────── validation ─────────────────────
        if grid.ndim != 2:
            raise ValueError("'grid' must be 2‑D")
        if selection.shape != grid.shape:
            raise ValueError("'selection' mask must match 'grid' shape")
        if selection.dtype != bool:
            raise TypeError("'selection' must be boolean")

        direction = direction.lower()
        mode      = mode.lower()
        if direction not in {"up", "right", "down", "left"}:
            raise ValueError(f"Unknown direction {direction!r}")
        if mode not in {"copy", "cut"}:
            raise ValueError("'mode' must be 'copy' or 'cut'")
        

        _background_color = find_background_colour(grid)
        all_obstacles = np.zeros_like(grid, bool)
        if obstacles:
            all_obstacles[grid != _background_color] = True

        def find_sub_masks(mask: np.ndarray, colour: 1) -> np.ndarray:
            _FOUR_CONN = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=bool)
            lbl, n = label(mask, structure=_FOUR_CONN)
            if n == 0:
                return np.zeros((0, *grid.shape), dtype=bool)
            return np.stack([(lbl == i) for i in range(1, n + 1)])

        # ───────────────────── inner core ─────────────────────
        def _slide_core(_grid: np.ndarray, _sel_mask: np.ndarray) -> None:
            """In‑place application of the sliding rules for *one* contiguous mask.

            Always works in *copy* mode to avoid clearing the original cells of
            other strips.  CUT handling is done once at the very end by the
            outer function.
            """

            # Coordinates of the sub‑selection
            sel_rows, sel_cols = np.nonzero(_sel_mask)
            if sel_rows.size == 0:
                return  # nothing to do
            sel_vals = out[sel_rows, sel_cols]

            n_rows, n_cols = _grid.shape

            # Direction vector
            drow, dcol = {
                "up":    (-1,  0),
                "right": ( 0,  1),
                "down":  ( 1,  0),
                "left":  ( 0, -1),
            }[direction]

            # Helper lambdas
            def in_bounds(rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
                return (
                    (rr >= 0) & (rr < n_rows) &
                    (cc >= 0) & (cc < n_cols)
                )

            def blocked(rr: np.ndarray, cc: np.ndarray, mask) -> np.ndarray:
                if not obstacles:
                    return np.zeros(rr.shape, bool)
                if fluid and mode == "cut":
                    # blocked if not background AND mask is False
                    return (_grid[rr, cc] != _background_color) & (~mask[rr, cc])
                return _grid[rr, cc] != _background_color

            # Step length (width or height of the current sub‑mask)
            step_len = (
                (np.ptp(sel_cols) + 1) if direction in {"left", "right"}
                else (np.ptp(sel_rows) + 1)
            )
            if mode == "cut" and fluid:
                step_len = 1
            #print('step_len: ', step_len)

            if continuous:  # ───────── tile repeatedly ─────────
                step_idx = 1
                while True:
                    r_off = step_idx * step_len * drow
                    c_off = step_idx * step_len * dcol
                    dest_rows = sel_rows + r_off
                    dest_cols = sel_cols + c_off

                    inside = in_bounds(dest_rows, dest_cols)
                    if not inside.any():
                        break  # nothing left inside grid

                    blk = blocked(dest_rows[inside], dest_cols[inside])

                    for i_in, i_sel in enumerate(np.flatnonzero(inside)):
                        if blk[i_in]:
                            continue
                        r = int(dest_rows[i_sel]); c = int(dest_cols[i_sel])
                        if _visited_global[r, c]:
                            continue
                        _grid[r, c] = int(sel_vals[i_sel])
                        _visited_global[r, c] = True

                    if obstacles and blk.all():
                        break  # whole tile blocked – stop early
                    step_idx += 1

            else:          # ─────── single placement ────────
                offset = 0
                while True:
                    trial_off = offset + 1
                    #print('trial_off: ', trial_off)
                    r_off = trial_off * drow
                    #print('r_off: ', r_off)
                    c_off = trial_off * dcol
                    #print('c_off: ', c_off)
                    dest_rows = sel_rows + r_off
                    dest_cols = sel_cols + c_off
                    #print('dest_rows: ', dest_rows)
                    #print('dest_cols: ', dest_cols)

                    if not in_bounds(dest_rows, dest_cols).all():
                        #print('would clip')
                        break  # would clip
                    if blocked(dest_rows, dest_cols, _sel_mask).any():
                        #print('Blocked:', blocked)
                        #print('blocked destination: ', blocked(dest_rows, dest_cols, _sel_mask))
                        #print('obstacle ahead')
                        break  # obstacle ahead
                    offset = trial_off
                    #print('offset: ', offset)

                if offset > 0:
                    r_off = offset * drow
                    c_off = offset * dcol
                    #print('r_off: ', r_off)
                    #print('c_off: ', c_off)
                    dest_rows = sel_rows + r_off
                    #print('dest_rows: ', dest_rows)
                    dest_cols = sel_cols + c_off
                    #print('dest_cols: ', dest_cols)
                    #print('sel_vals: ', sel_vals)
                    _grid[dest_rows, dest_cols] = sel_vals
                    #print('grid: ', _grid)
                    _visited_global[dest_rows, dest_cols] = True
                    _visited_this_round = np.zeros_like(_sel_mask, bool)
                    _visited_this_round[dest_rows, dest_cols] = True
                    #print('visited_global: ', _visited_global)

                    if mode == "cut":
                        #print('mode == cut')
                        #print('sel_mask: ', _sel_mask)
                        #print('visited_global: ', _visited_this_round)
                        _delete_mask = _sel_mask & ~_visited_this_round
                        #print('delete_mask: ', _delete_mask)
                        _grid[_delete_mask] = _background_color

        # ──────────────────── main body ───────────────────────
        _grid_copy_source = grid.copy()  # reference for original colours
        out = grid.copy()
        _visited_global = np.zeros_like(selection, bool)

        if not fluid:
            _slide_core(out, selection)
        else:
            if direction in {"up", "down"}:   # treat each *column* independently
                #print('np.flatnonzero(selection.any(axis=0)): ', np.flatnonzero(selection.any(axis=0)))
                for col_idx in np.flatnonzero(selection.any(axis=0)):
                    #print('col_idx: ', col_idx)
                    sub_mask = selection & (np.arange(grid.shape[1]) == col_idx)
                    if superfluid:
                        sub_masks = find_sub_masks(sub_mask, 1)
                        #print('sub_mask: ', sub_mask)
                        #print('sub_masks: ', sub_masks)
                        for sub_mask in sub_masks:
                            #print('visited_global & np.arange(grid.shape[1]) == col_idx: ', _visited_global & (np.arange(grid.shape[1]) == col_idx))
                            connected_sub_mask = sub_mask | _visited_global & (np.arange(grid.shape[1]) == col_idx)
                            #print('connected_sub_mask: ', connected_sub_mask)
                            _slide_core(out, connected_sub_mask)
                            #print('out: \n', out)
                            #print('')
                    else:
                        _slide_core(out, sub_mask)
            else:                               # left / right ⇒ independent rows
                for row_idx in np.flatnonzero(selection.any(axis=1)):
                    sub_mask = selection & (np.arange(grid.shape[0])[:, None] == row_idx)
                    if superfluid:
                        sub_masks = find_sub_masks(sub_mask, 1)
                        for sub_mask in sub_masks:
                            _slide_core(out, sub_mask | _visited_global & (np.arange(grid.shape[0])[:, None] == row_idx))
                    else:
                        _slide_core(out, sub_mask)

        # ───────────── CUT mode (done once!) ─────────────
        if mode == "cut":
            pass
            #mask = selection & ~_visited_global
            #out[mask] = _background_color

        return out
        
    move_down_superfluid, move_down_superfluid.__name__ = partial(slide_old, direction="down", mode="cut", continuous=False, fluid=True), "move_down_superfluid"
