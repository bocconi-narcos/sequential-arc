"""
ARC Gymnasium environment that plays nicely with the new `ARCActionSpace`.

Highlights
──────────
* works with **either** `mode="factorized"` (Dict) or `mode="joint"` (Discrete)
* keeps an internal BFS‐style frontier (deque) so we can explore several
  alternative grids per step
* reward is overlap‑based + small step / shape penalties (same logic as before
  but refactored)
* independent RNG via `np.random.default_rng`
"""

from __future__ import annotations

import json
import copy
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from action_space import ARCActionSpace
from dsl.utils.padding import pad_grid, unpad_grid

# ───────────────────────────────────────────────────────────────────── #
#  Render (human mode only)
# ───────────────────────────────────────────────────────────────────── #
import matplotlib.pyplot as plt
from matplotlib import colors

# canonical ARC colour‑map  (‑1 is white)
_ARC_CMAP = colors.ListedColormap(
    [
        "#FFFFFF", "#000000", "#0074D9", "#FF4136", "#2ECC40",
        "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
        "#39CCCC"  # New distant color added
    ]
)


# ───────────────────────────────────────────────────────────────────── #
# Utility functions
# ───────────────────────────────────────────────────────────────────── #
def _maximum_overlap(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    Return the highest fraction of cells that match when `arr2` is slid over
    `arr1` in every possible position (4‑connected translations, no rotations).

    The function is intentionally simple & robust; performance is fine for the
    ≤ 30×30 grids typical of ARC.  Vectorised rewrite can be added later.
    """
    h1, w1 = arr1.shape
    h2, w2 = arr2.shape

    # fast path – equal shapes
    if (h1, w1) == (h2, w2):
        return float(np.mean(arr1 == arr2))

    best = 0
    for dy in range(-h2 + 1, h1):
        r1s = max(0,  dy)
        r1e = min(h1, dy + h2)
        r2s = max(0, -dy)
        r2e = r2s + (r1e - r1s)

        for dx in range(-w2 + 1, w1):
            c1s = max(0,  dx)
            c1e = min(w1, dx + w2)
            c2s = max(0, -dx)
            c2e = c2s + (c1e - c1s)

            overlap = np.sum(arr1[r1s:r1e, c1s:c1e] ==
                             arr2[r2s:r2e, c2s:c2e])
            if overlap > best:
                best = overlap

    return best / arr2.size


def _extract_grids(raw: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (input_grid, output_grid) as NumPy arrays."""
    try: 
        return np.array(raw["input"], dtype=np.int8), np.array(raw["output"], dtype=np.int8)
    except:
        print(f"Failed to extract grids from {raw}")
        raise ValueError("Invalid input/output grids in JSON data.")


# ───────────────────────────────────────────────────────────────────── #
# Main environment
# ───────────────────────────────────────────────────────────────────── #
class ARCEnv(gym.Env):
    """
    A Gymnasium environment for the Abstraction and Reasoning Corpus (ARC).

    The environment loads ARC challenges from JSON files. Each step involves applying
    a three-part hierarchical action (colour, selection, transformation) to an
    input grid. The goal is to transform the input grid into the target grid.

    Observation Space:
        A 2-channel NumPy array of shape (canvas_size, canvas_size, 2).
        - Channel 0: The current working grid.
        - Channel 1: The target grid for the current challenge.
        Values represent colours (0-9, typically).

    Action Space:
        An `ARCActionSpace` instance, which can be in "factorized" (Dict) or
        "joint" (Discrete) mode. This space defines the available DSL functions
        for colouring, selecting, and transforming grid regions.

    Reward:
        The reward is based on the change in maximum overlap between the current
        grid and the target grid. Penalties are applied for no-change actions,
        mismatched shapes, and producing an all-one-colour grid (truncation).
        A bonus is given for successfully solving the challenge.

    Internal Mechanics:
        The environment does not currently use a BFS-style frontier as mentioned
        in the module docstring, but processes one action at a time on the current
        grid state. The `_frontier` and `_info_q` are initialized but not used
        in the current `step` logic. (Note: This might be a point for future review
        or refactoring if the frontier logic is intended to be active).
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}  # reserved

    # ------------------------------- #
    # Construction
    # ------------------------------- #
    def __init__(
        self,
        challenges_json: str | Path,
        solutions_json: str | Path,
        *,
        action_space: ARCActionSpace,
        canvas_size: int = 30,
        step_penalty: int = 1,
        shape_penalty: int = 1,
        no_change_penalty: int = 5,
        trunc_penalty: int = 100,
        completion_bonus: int = 25,
        max_branch: int = 1,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the ARC Environment.

        Args:
            challenges_json: Path to the JSON file containing ARC challenge definitions.
            solutions_json: Path to the JSON file containing ARC solution definitions
                            (used for evaluation, not directly during training steps).
            action_space: An initialized `ARCActionSpace` instance.
            canvas_size: The height and width of the observation grid. Grids smaller
                         than this will be padded. Defaults to 30.
            step_penalty: Penalty for actions that improve overlap by <= 0 but don't
                          result in no change. Defaults to 1.
            shape_penalty: Additional penalty if the grid shape doesn't match the target
                           shape and overlap improvement is <=0. Defaults to 1.
            no_change_penalty: Penalty for actions that result in no change to the grid.
                               Defaults to 5.
            trunc_penalty: Penalty for producing an all-one-colour grid if it's not
                           the solution. Defaults to 100.
            completion_bonus: Bonus reward for successfully solving the challenge.
                              Defaults to 25.
            max_branch: (Currently unused) Intended for BFS-style exploration.
                        Defaults to 1.
            seed: Optional seed for the environment's random number generator.
                  Defaults to None.
        """
        super().__init__()

        # Canvas size
        self._canvas_shape = (canvas_size, canvas_size)

        # RNG
        self._rng = np.random.default_rng(seed)

        # Load challenge corpus
        self._challenges: Dict[str, Any] = json.load(open(challenges_json))
        self._solutions = json.load(open(solutions_json))
        self._keys: Tuple[str, ...] = tuple(self._challenges)

        # Spaces
        self.action_space: ARCActionSpace = action_space
        H = W = canvas_size
        self.observation_space = gym.spaces.Box(
            low=0, high=9, shape=(H, W, 2), dtype=np.int8
        )

        # Reward hyper-parameters
        self._step_penalty = step_penalty
        self._shape_penalty = shape_penalty
        self._no_change_penalty = no_change_penalty
        self._trunc_penalty = trunc_penalty
        self._bonus = completion_bonus
        self._max_branch = max_branch

        # runtime buffers
        self._frontier: deque[Tuple[np.ndarray, np.ndarray]] = deque()
        self._info_q: deque[Dict] = deque()

        # exposed vars
        self.state: np.ndarray | None = None
        self.info: Dict[str, Any] | None = None # Information about the current episode
        self._last_render: Dict[str, Any] = {} # Data for the last rendered step

    # ------------------------------- #
    # Private helpers
    # ------------------------------- #
    def _sample_challenge(
        self,
        key: str | None = None,
        min_examples: int | None = None,
        index: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Sample a (input_grid, output_grid, challenge_key) tuple from the loaded challenges.

        This method selects a challenge based on the provided criteria. It can pick a
        random challenge, a specific challenge by its key, or a specific
        input-output pair within a challenge by its index.

        Args:
            key: If provided, forces the selection of the challenge with this key.
                 An error is raised if the key is not found or if the challenge
                 does not meet `min_examples` criteria.
            min_examples: If provided, only challenges with at least this many
                          training examples are considered for random selection.
            index: If provided along with `key`, selects the specific training
                   example at this index within the chosen challenge. An error
                   is raised if the index is out of bounds.

        Returns:
            A tuple containing:
                - input_grid (np.ndarray): The input grid for the sampled example.
                - output_grid (np.ndarray): The target output grid for the sampled example.
                - challenge_key (str): The key of the challenge from which the example was sampled.

        Raises:
            ValueError: If no challenges meet `min_examples`, or if a specified
                        challenge does not meet `min_examples`, or if input/output
                        grids are invalid in the JSON data.
            KeyError: If a specified `key` is not found in the loaded challenges.
            IndexError: If a specified `index` is out of bounds for the selected challenge.
        """
        # 1) filter by min_examples
        viable_keys = self._keys
        if min_examples is not None:
            viable_keys = [
                k for k in viable_keys
                if len(self._challenges[k]["train"]) >= min_examples
            ]
            if not viable_keys:
                raise ValueError(
                    f"No challenges have at least {min_examples} examples."
                )

        # 2) resolve key
        if key is None:
            key = self._rng.choice(viable_keys)
        else:
            if key not in self._challenges:
                raise KeyError(f"Challenge key '{key}' not found.")
            if min_examples is not None and len(self._challenges[key]["train"]) < min_examples:
                raise ValueError(
                    f"Challenge '{key}' has only "
                    f"{len(self._challenges[key]['train'])} examples,"
                    f" fewer than required {min_examples}."
                )

        # 3) pick the specific or a random pair
        train_set = self._challenges[key]["train"]
        if index is not None:
            if not (0 <= index < len(train_set)):
                raise IndexError(
                    f"Index {index} out of range for challenge '{key}'"
                    f" (size {len(train_set)})."
                )
            pair = train_set[index]
        else:
            pair = self._rng.choice(train_set)

        inp, out = _extract_grids(pair)
        return inp, out, key

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None
    ):
        """
        Reset env.  options may include:
          - 'key': a specific challenge key
          - 'min_examples': require at least that many train examples
          - 'index': pick the train example at that index
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._frontier.clear()
        self._info_q.clear()

        forced_key = None
        min_examples = None
        index = None
        if options:
            forced_key   = options.get("key", None)
            min_examples = options.get("min_examples", None)
            index        = options.get("index", None)

        inp, out, key = self._sample_challenge(
            key=forced_key,
            min_examples=min_examples,
            index=index
        )

        # build padded state: channel 0 = input, channel 1 = target
        grid = np.zeros(
            (*self.observation_space.shape[:2], 2),
            dtype=np.int8
        )
        grid[..., 0] = pad_grid(inp, self._canvas_shape)
        grid[..., 1] = pad_grid(out, self._canvas_shape)

        self.state = grid
        self.info = {
            "key":         key,
            "actions":     [],
            "action_desc": [],
            "num_actions": 0,
            "solved":      False,
        }
        return self.state, {}
    
    def render(self, *, mode: str = "human"):
        """
        Visualises **before**, **mask**, **after**, **target** and (inset) the
        **chosen colour**.  Call *after* step().
        """
        if mode != "human":
            raise NotImplementedError("Only mode='human' is supported.")
        if not hasattr(self, "_last_render"):
            print("Nothing to render yet — run step first.")
            return

        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors

        # ── ARC colour map ─────────────────────────────────────────── #
        cmap = _ARC_CMAP

        data    = self._last_render
        before  = data["before"]
        after   = data["after"]
        target  = data["target"]
        mask    = data["mask"]
        colour  = data["colour"]
        desc    = data["action_str"]
        reward  = data["reward"]
        key     = self.info.get("key", "-")            #  ←  challenge key

        # ── layout: 1×4 grid … colour is an inset above the mask ───── #
        fig, axes = plt.subplots(1, 4, figsize=(12, 5))
        titles = ["Before", "Mask", "After", "Target"]

        def _show_grid(ax, grid, title):
            ax.imshow(grid, cmap=cmap, vmin=-1, vmax=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9)

        _show_grid(axes[0], before,  titles[0])

        # ── MASK panel ─────────────────────────────────────────────────── #
        axes[1].imshow(mask.astype(int), cmap="gray", vmin=0, vmax=1)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        # --- title + colour swatch packed side-by-side ------------------- #
        from matplotlib.offsetbox import TextArea, DrawingArea, HPacker, AnchoredOffsetbox
        import matplotlib.patches as mpatches

        # 1) the title text
        title_box = TextArea("Mask given selected color:", textprops=dict(size=9))

        # 2) the little square showing the selected colour
        side = 12                       # pixels; change if you want it bigger/smaller
        swatch_da = DrawingArea(side, side, 0, 0)
        swatch_da.add_artist(
            mpatches.Rectangle((0, 0), side, side,
                            facecolor=cmap(colour+1), edgecolor="0.3", linewidth=0.5)
        )

        # 3) pack them horizontally with a tiny gap
        packed = HPacker(children=[title_box, swatch_da],
                        align="center", pad=0, sep=4)

        # 4) anchor the whole pack centred over the axes
        anch = AnchoredOffsetbox(loc='upper center',
                                child=packed,
                                pad=0.,  frameon=False,
                                bbox_to_anchor=(0.5, 1.101),   # x, y in axes coords
                                bbox_transform=axes[1].transAxes,
                                borderpad=0.)
        axes[1].add_artist(anch)


        _show_grid(axes[2], after,   titles[2])
        _show_grid(axes[3], target,  titles[3])

        # ── global title (action, reward and challenge key) ─────────── #
        fig.suptitle(f"{desc}   |   Reward {reward:+.1f}   |   Challenge {key}",
                    fontsize=11, y=0.97)

        plt.tight_layout()
        plt.show()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.

        Applies the given action to the current grid state, calculates the reward,
        and determines if the episode has terminated or truncated.

        Args:
            action: An action compatible with `self.action_space`. This will be
                    decoded into colour, selection, and transformation functions.

        Returns:
            A tuple (observation, reward, terminated, truncated, info):
            - observation (np.ndarray): The new state of the environment, with the
                                      transformed grid in channel 0 and the target
                                      grid in channel 1.
            - reward (float): The reward achieved by the action.
            - terminated (bool): Whether the episode has ended successfully (target reached).
            - truncated (bool): Whether the episode has ended due to a condition like
                                producing an all-one-colour grid (and not solving).
            - info (Dict[str, Any]): A dictionary containing auxiliary diagnostic
                                     information (e.g., 'key', 'actions', 'solved').

        Raises:
            AssertionError: If `reset()` has not been called before `step()`.
        """
        assert self.state is not None, "Call reset() before step()."
        prev_inp = unpad_grid(self.state[..., 0])
        target   = unpad_grid(self.state[..., 1])

        # ---------- decode the action ------------------------------------ #
        colour_fn, select_fn, transform_fn = self.action_space.decode(action)
        colour    = colour_fn(grid = prev_inp)
        sel_mask  = select_fn(prev_inp, colour)
        next_grid = transform_fn(prev_inp, sel_mask)

        # ---------- Reward calculation ------------------------------------ #
        # Penalize if the grid hasn't changed
        if next_grid.shape == prev_inp.shape and np.array_equal(next_grid, prev_inp):
            reward = -self._no_change_penalty
        else:
            # Calculate reward based on change in overlap with target
            prev_overlap = _maximum_overlap(prev_inp, target)
            curr_overlap = _maximum_overlap(next_grid, target)
            reward = (curr_overlap - prev_overlap) * 100  # Scale to be more significant

            # Penalize if no improvement or negative improvement
            if reward <= 0:
                reward -= self._step_penalty
                # Additional penalty if shape also doesn't match target
                if next_grid.shape != target.shape:
                    reward -= self._shape_penalty

        # ---------- Termination and Truncation conditions ----------------- #
        # Terminated if the current grid matches the target grid
        terminated = bool(next_grid.shape == target.shape and np.array_equal(next_grid, target))
        # Truncated if the grid becomes a single solid color (and is not the solution)
        truncated  = bool(len(np.unique(next_grid)) == 1 and not terminated)

        if terminated:
            reward += self._bonus  # Add completion bonus
        if truncated: # No need to check 'and not terminated' due to earlier logic
            reward -= self._trunc_penalty # Penalize for collapsing to single color

        # ---------- Bookkeeping ----------------------------------------- #
        assert self.info is not None # Should be initialized in reset
        self.info["actions"].append(action)
        self.info["action_desc"].append(self.action_space.action_to_str(action))
        self.info["num_actions"] += 1
        self.info["solved"] = terminated

        # ---------- build new observation -------------------------------- #
        obs = np.zeros_like(self.state)
        obs[..., 0] = pad_grid(next_grid, self._canvas_shape)
        obs[..., 1] = self.state[..., 1]        # keep target
        self.state = obs

        # ---------- store data for render() ----------------------------- #
        self._last_render = {
            "before":   prev_inp,
            "after":    next_grid,
            "target":   target,
            "mask":     sel_mask,
            "colour":   colour,
            "action_str": self.action_space.action_to_str(action),
            "reward":   reward,
        }

        return obs, reward, terminated, truncated, copy.deepcopy(self.info)
    
    # ───────────────────────────────────────────────────────────────── #
    #  Batch-testing a fixed action sequence
    # ───────────────────────────────────────────────────────────────── #
    def evaluate_sequence(
        self,
        challenge_key: str,
        actions: list[int] | np.ndarray,
        *,
        render: bool = True,
        exclude: Iterable[int] = (),
        include: Iterable[int] | None = None,
        stepwise: bool = False,
    ):
        """
        Execute `actions` on *all* grids of `challenge_key`.

        Parameters
        ----------
        challenge_key : str
            Key present in ``self._challenges``.
        actions : list[int] | np.ndarray
            Sequence of actions, already encoded for ``self.action_space``.
        render : bool, default True
            If True, draw a single figure that stacks
            **before → after → target** for every grid.

        Returns
        -------
        results : list[dict]
            One entry per grid with keys::
                before   – original input grid  (unpadded)
                after    – grid after all actions (unpadded)
                target   – ground-truth output grid (unpadded)
                reward   – cumulative reward over the sequence
                solved   – bool, whether final grid == target
                pair_idx – position within the challenge set
        """
        exclude = set(exclude)
        include = None if include is None else set(include)

        if challenge_key not in self._challenges:
            raise KeyError(f"Unknown challenge key: {challenge_key}")
        
        # gather every (input, output) pair – train *and* test
        pairs = list(self._challenges[challenge_key]["train"])

        # assert each element is a dict with "input" and "output" keys
        for pair in pairs:
            if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
                raise ValueError(f"Invalid pair format: {pair}")
        #pairs += list(self._challenges[challenge_key].get("test", []))

        results: list[dict] = []

        # save & restore RNG state so we do not disturb reproducibility
        rng_state = self._rng.__getstate__()

        for pair_idx, pair in enumerate(pairs):
            if pair_idx in exclude: # skip excluded pairs
                continue
            if include is not None and pair_idx not in include:
                continue
            inp, tgt = _extract_grids(pair)

            # --- initialise env state to this pair ---------------------- #
            grid = np.zeros((*self.observation_space.shape[:2], 2), dtype=np.int8)
            grid[..., 0] = pad_grid(inp, self._canvas_shape)
            grid[..., 1] = pad_grid(tgt, self._canvas_shape)
            self.state = grid
            self.info = {
                "key": challenge_key,
                "actions": [],
                "action_desc": [],
                "num_actions": 0,
                "solved": False,
            }

            # --- roll through the fixed sequence ----------------------- #
            terminated = truncated = False
            total_reward = 0.0
            for a in actions:
                if terminated or truncated:
                    break
                _, r, terminated, truncated, _ = self.step(a)
                total_reward += r
                if stepwise:
                    self.render()

            results.append(
                dict(
                    before=inp,
                    after=unpad_grid(self.state[..., 0]),
                    target=tgt,
                    reward=total_reward,
                    solved=terminated,
                    pair_idx=pair_idx,
                )
            )

        # restore RNG
        self._rng.__setstate__(rng_state)

        # ---------------------------------------------------------------- #
        # Optional summary figure
        # ---------------------------------------------------------------- #
        if render:
            import matplotlib.pyplot as plt

            n = len(results)
            fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
            if n == 1:          # ensure 2-D indexability
                axes = axes.reshape(1, 3)

            for row, res in enumerate(results):
                for col, (title, img) in enumerate(
                    [("Before", res["before"]),
                     ("After",  res["after"]),
                     ("Target", res["target"])]
                ):
                    ax = axes[row, col]
                    ax.imshow(img, cmap=_ARC_CMAP, vmin=-1, vmax=10)
                    ax.set_xticks([]); ax.set_yticks([])
                    if row == 0:
                        ax.set_title(title, fontsize=9)

                # annotate row with reward / success
                axes[row, 1].set_ylabel(
                    f"Pair {row} | R={res['reward']:+.1f}"
                    f" | {'✓' if res['solved'] else '✗'}",
                    rotation=0, ha="right", va="center", fontsize=8
                )

            plt.tight_layout()

            # save the figure to a file
            fig.savefig(f"solvers/renders/solver_{challenge_key}.png", dpi=150, bbox_inches="tight")

            # check for success in all pairs
            all_solved = all(res["solved"] for res in results)
            if all_solved:
                pass
            else:
                print("Some pairs were not solved. Check the output.")
                for res in results:
                    if not res["solved"]:
                        print(f"Pair {res['pair_idx']} failed with reward {res['reward']:.1f}")

        return results

