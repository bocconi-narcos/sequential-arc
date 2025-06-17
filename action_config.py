"""
Action‑space *configuration* ― lists of which DSL operators (and parameters)
should be exposed in each of the three sub‑spaces.

Each entry is a mapping
    {"name": <method‑name‑str>, "kwargs": {<param>: <value>, ...}}

The strings are resolved with getattr() against the DSL classes, so no
imports from this file leak into user code.
"""

from __future__ import annotations
from typing import Dict, List

Preset = Dict[str, List[dict]]          # alias for readability

# ─────────────────────────────────────────────────────────────────── #
# Define as many presets as you like
# ─────────────────────────────────────────────────────────────────── #
PRESETS: Dict[str, Preset] = {

    # -------------------------------------------------------------- #
    #  Minimal – one operator per sub‑space (great for unit tests)
    # -------------------------------------------------------------- #
    "minimal": {
        "colour": [
            {"name": "most_common", "kwargs": {}},
        ],
        "selection": [
            {"name": "all_cells",   "kwargs": {}},
        ],
        "transform": [
            {"name": "identity",    "kwargs": {}},
        ],
    },

        "test": {
        "colour": [
            {"name": "most_independent_cells", "kwargs": {}},
        ],
        "selection": [
            {"name": "colour",       "kwargs": {}},
        ],
        "transform": [
            {"name": "move_down_superfluid", "kwargs": {}},
        ],
    },

    # -------------------------------------------------------------- #
    #  Default – reasonably expressive but still lightweight
    # -------------------------------------------------------------- #
    "default": {
        "colour": [
            {"name": "most_common", "kwargs": {}},
            {"name": "least_common",     "kwargs": {}},
            {"name": "second_most_common",  "kwargs": {}},
            {"name": "third_most_common",  "kwargs": {}},
            {"name": "fourth_most_common", "kwargs": {}},
            {"name": "fifth_most_common",  "kwargs": {}},
            {"name": "sixth_most_common",  "kwargs": {}},
            {"name": "seventh_most_common", "kwargs": {}},
            {"name": "eighth_most_common",  "kwargs": {}},
            {"name": "most_independent_cells", "kwargs": {}},
            {"name": "second_most_independent_cells", "kwargs": {}},
            {"name": "third_most_independent_cells", "kwargs": {}},
            {"name": "colour_0", "kwargs": {}},
            {"name": "colour_1", "kwargs": {}},
            {"name": "colour_2", "kwargs": {}},
            {"name": "colour_3", "kwargs": {}},
            {"name": "colour_4", "kwargs": {}},
            {"name": "colour_5", "kwargs": {}},
            {"name": "colour_6", "kwargs": {}},
            {"name": "colour_7", "kwargs": {}},
            {"name": "colour_8", "kwargs": {}},
            {"name": "colour_9", "kwargs": {}},
            {"name": "colour_10", "kwargs": {}},

        ],
        "selection": [
            {"name": "all_cells",   "kwargs": {}},
            {"name": "colour",       "kwargs": {}},
            {"name": "contact4_2", "kwargs": {}},
            {"name": "independent_cells_4", "kwargs": {}},
            {"name": "outer_border8", "kwargs": {}},
            {"name": "grid_border", "kwargs": {}},
        ],
        "transform": [
            {"name": "rotate_180",     "kwargs": {}},
            {"name": "rotate_90",      "kwargs": {}},
            {"name": "new_colour_0", "kwargs": {}},
            {"name": "new_colour_1", "kwargs": {}},
            {"name": "new_colour_2", "kwargs": {}},
            {"name": "new_colour_3", "kwargs": {}},
            {"name": "new_colour_4", "kwargs": {}},
            {"name": "new_colour_5", "kwargs": {}},
            {"name": "new_colour_6", "kwargs": {}},
            {"name": "new_colour_7", "kwargs": {}},
            {"name": "new_colour_8", "kwargs": {}},
            {"name": "new_colour_9", "kwargs": {}},
            {"name": "new_colour_10", "kwargs": {}},
            {"name": "move_down_superfluid", "kwargs": {}},
            {"name": "flip_vertical", "kwargs": {}},
            {"name": "flip_horizontal", "kwargs": {}},
            {"name": "background_colour", "kwargs": {}},
            {"name": "invert_colors", "kwargs": {}},
        ],
    },
}
