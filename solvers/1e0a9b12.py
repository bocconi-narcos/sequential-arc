from typing import List
from action_space import ARCActionSpace

EXCLUDE = {
    44, # Ambiguous solution, could be both with yelllow or grey
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
        space.encode(colour='second_most_common', selection="colour", transform="move_down_superfluid", key=key),
        space.encode(colour='third_most_common', selection="colour", transform="move_down_superfluid", key=key),
        space.encode(colour='fourth_most_common', selection="colour", transform="move_down_superfluid", key=key),
        space.encode(colour='fifth_most_common', selection="colour", transform="move_down_superfluid", key=key),
        space.encode(colour='sixth_most_common', selection="colour", transform="move_down_superfluid", key=key),
        space.encode(colour='seventh_most_common', selection="colour", transform="move_down_superfluid", key=key),
    ]
