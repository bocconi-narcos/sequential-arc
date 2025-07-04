# solvers/3aa6fb7a.py

from typing import List
from action_space import ARCActionSpace
EXCLUDE = {
}

def solve(space: ARCActionSpace) -> List[int]:
    """
    Return the exact sequence of encoded actions (ints) that solve challenge 3aa6fb7a.
    Use `space.encode(colour, select, transform)` to keep it human–readable.
    """
    # key is the name of the file without the extension
    file_name = __file__.split("/")[-1]
    key = file_name.split(".")[0]

    return [
        space.encode(colour='second_most_common', 
                     selection="contact4_2", 
                     transform="new_colour_1", 
                     key=key),
    ]




