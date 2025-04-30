"""
arc_env/action_space.py

Hierarchical / joint Gymnasium action space for ARC with human‑readable
pretty‑printing.

Key API
───────
ARCActionSpace(preset="default", mode="factorized")
    • mode="factorized"  → Dict("colour", "selection", "transform")
    • mode="joint"       → Discrete(C×S×T)

.decode(action)          -> (colour_fn, selection_fn, transform_fn)
.action_to_str(action)   -> str
"""

from __future__ import annotations
from functools import partial
from types import FunctionType
from typing import Callable, Dict, List, Tuple, Union

import gymnasium as gym

from dsl.colour import ColorSelector
from dsl.select       import GridSelector
from dsl.transform    import GridTransformer
from action_config    import PRESETS


# ───────────────────────────────────────────────────────────────────── #
# Helper: build library from config
# ───────────────────────────────────────────────────────────────────── #
def _build_library(preset: str, category: str,
                   obj,                       # instance of DSL class
                   ) -> Tuple[Tuple[Callable, ...], int]:
    cfg = PRESETS[preset][category]
    fns: List[Callable] = []
    for entry in cfg:
        base = getattr(obj, entry["name"])
        fns.append(
            partial(base, **entry["kwargs"]) if entry.get("kwargs") else base
        )
    return tuple(fns), len(fns)


# ───────────────────────────────────────────────────────────────────── #
# The action‑space class
# ───────────────────────────────────────────────────────────────────── #
class ARCActionSpace(gym.Space):
    """
    If *mode* == "factorized": behaves exactly like the previous Dict version
    (three Discrete sub‑spaces).  Perfect when your agent has **three heads**
    that learn independent distributions.

    If *mode* == "joint": exposes one gigantic Discrete space of size
    C × S × T.  Good for single‑head agents / tabular baselines.
    """

    metadata = {"render.modes": []}

    def __init__(self, *, preset: str = "default", mode: str = "factorized") -> None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'.  Available: {list(PRESETS)}")
        if mode not in ("factorized", "joint"):
            raise ValueError("mode must be 'factorized' or 'joint'")

        self.mode = mode

        # instantiate DSL singletons
        self._cs  = ColorSelector()
        self._sel = GridSelector()
        self._tr  = GridTransformer()

        # build libraries
        self._lib_c, self._C = _build_library(preset, "colour",    self._cs)
        self._lib_s, self._S = _build_library(preset, "selection", self._sel)
        self._lib_t, self._T = _build_library(preset, "transform", self._tr)

        # ------------------------------------------------------------------ #
        # choose underlying Gym space
        # ------------------------------------------------------------------ #
        if mode == "factorized":
            self.space = gym.spaces.Dict({
                "colour":    gym.spaces.Discrete(self._C),
                "selection": gym.spaces.Discrete(self._S),
                "transform": gym.spaces.Discrete(self._T),
            })
        else:                                 # "joint"
            self._joint_size = self._C * self._S * self._T
            self.space = gym.spaces.Discrete(self._joint_size)

        super().__init__(self.space.shape if hasattr(self.space, "shape") else (),
                         self.space.dtype if hasattr(self.space, "dtype") else None)

    # ------------------------------------------------------------------ #
    # Gymnasium API proxy
    # ------------------------------------------------------------------ #
    def sample(self, mask: Dict | None = None):
        return self.space.sample(mask)

    def contains(self, x) -> bool:
        return self.space.contains(x)

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def _decode_joint(self, idx: int):
        c = idx // (self._S * self._T)
        s = (idx // self._T) % self._S
        t = idx % self._T
        return c, s, t

    def decode(self, action):
        """Return the triple (colour_fn, selection_fn, transform_fn)."""

        if self.mode == "factorized":
            c, s, t = action["colour"], action["selection"], action["transform"]
        else:                                 # joint
            c, s, t = self._decode_joint(int(action))
        return self._lib_c[c], self._lib_s[s], self._lib_t[t]
    
    def encode(self, colour: str, selection: str, transform: str, key=None) -> Union[Dict[str, int], int]:
        """Encode an action given as strings into the internal integer representation.

        Args:
            colour (str): Colour function name.
            select (str): Selection function name.
            transform (str): Transformation function name.

        Returns:
            In 'factorized' mode, a dict mapping each sub-space to its integer index.
            In 'joint' mode, a single integer combining all three indices.

        Raises:
            ValueError: if any of the provided function strings is not available.
        """

        def find_index(lib: List[Callable], name: str) -> int:
            for i, fn in enumerate(lib):
                fn_str = ARCActionSpace._fn_to_str(fn)
                # allow matching either the full qualname or just the method name
                simple_str = fn_str.split('.', 1)[-1]
                # remove the '()' suffix for comparison
                if simple_str.endswith('()'):
                    simple_str = simple_str[:-2]
                if name == fn_str or name == simple_str:
                    return i
            avail = [ARCActionSpace._fn_to_str(fn) for fn in lib]
            # check if key is None
            if key is not None:
                raise ValueError(f"Unknown action '{name}'. Available: {avail}.\n\nCheck solver for challenge_key f'{key}'")
            else:
                raise ValueError(f"Unknown action '{name}'. Available: {avail}")

        c_name = colour
        s_name = selection
        t_name = transform

        c_idx = find_index(self._lib_c, c_name)
        s_idx = find_index(self._lib_s, s_name)
        t_idx = find_index(self._lib_t, t_name)

        if self.mode == "factorized":
            return {"colour": c_idx, "selection": s_idx, "transform": t_idx}
        return c_idx * self._S * self._T + s_idx * self._T + t_idx


    # ------------------------------------------------------------------ #
    # Pretty‑printer
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fn_to_str(fn: Callable) -> str:
        """Return 'name(param=val, ...)' for a callable (partial or method)."""

        # 1) If someone has explicitly set __name__ on the partial, use that.
        explicit = getattr(fn, "__name__", None)
        if explicit and not isinstance(fn, FunctionType):
            # if it's a partial with an overridden __name__, just show that
            return explicit + "()"

        # 2) Next, handle a normal partial without an explicit name
        if isinstance(fn, partial):
            name = fn.func.__name__
            params = ", ".join(f"{k}={v}" for k, v in fn.keywords.items())
            return f"{name}({params})" if params else name + "()"

        # 3) Plain functions
        if isinstance(fn, FunctionType):
            return fn.__name__ + "()"

        # 4) Bound methods, other callables
        return fn.__qualname__ + "()"

    def action_to_str(self, action) -> str:
        c_fn, s_fn, t_fn = self.decode(action)
        return (f"[Colour] {self._fn_to_str(c_fn)}  →  "
                f"[Select] {self._fn_to_str(s_fn)}  →  "
                f"[Transform] {self._fn_to_str(t_fn)}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def sizes(self) -> Dict[str, int]:
        return {"colour": self._C, "selection": self._S, "transform": self._T}

    # delegate other special methods to inner space
    def __getattr__(self, item):        # noqa: D401
        return getattr(self.space, item)