from typing import List
from action_space import ARCActionSpace
EXCLUDE = {
}

def solve(space: ARCActionSpace) -> List[int]:
    """
    Return the exact sequence of encoded actions (ints) that solve challenge 1ef4f85a.
    Use `space.encode(colour, select, transform)` (or whatever encode API you have)
    to keep it human–readable.
    """
    file_name = __file__.split("/")[-1]
    key = file_name.split(".")[0]
    return [
        # example:
        space.encode(colour='most_common', selection="all_cells", transform="invert_colors", key=key),
        space.encode(colour='colour_5', selection='colour', transform='new_colour_0', key=key),
    ]
