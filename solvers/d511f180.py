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
        space.encode(colour='colour_5', selection='colour', transform='new_colour_10', key=key),
        space.encode(colour='colour_8', selection='colour', transform='new_colour_5', key=key),
        space.encode(colour='colour_10', selection='colour', transform='new_colour_8', key=key),
    ]