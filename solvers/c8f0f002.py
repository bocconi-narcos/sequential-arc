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
        space.encode(colour='colour_7', selection="colour", transform="new_colour_5", key=key),
    ]
