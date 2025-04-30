import pkgutil, importlib
from typing import Callable, Dict, List, Set

SOLVERS:   Dict[str, Callable]  = {}
EXCLUDES:  Dict[str, Set[int]]  = {}

for _, modname, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{modname}")

    # --- mandatory ---------------------------------------------------- #
    if hasattr(module, "solve"):
        SOLVERS[modname] = module.solve
    else:
        continue

    # --- optional ----------------------------------------------------- #
    excl = getattr(module, "EXCLUDE", set())
    EXCLUDES[modname] = set(excl)

# helpers -------------------------------------------------------------- #
def list_keys() -> List[str]:
    return list(SOLVERS)

def get_actions(key: str, space) -> List[int]:
    if key not in SOLVERS:
        raise KeyError(f"No solver for '{key}'.")
    return SOLVERS[key](space)

def get_exclude(key: str) -> Set[int]:
    return EXCLUDES.get(key, set())
