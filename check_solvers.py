"""
Verify every solver in `solvers/` really solves its challenge.
"""

import argparse
import sys
from pathlib import Path

from solvers import SOLVERS, get_actions, list_keys, get_exclude

# your existing imports
from action_space import ARCActionSpace
from env import ARCEnv

# adjust paths
DATA_DIR = Path(__file__).resolve().parent / "data"
CHALLENGES_JSON = DATA_DIR / "challenges.json"
SOLUTIONS_JSON  = DATA_DIR / "solutions.json"

def build_env(seed: int | None = 0) -> ARCEnv:
    space = ARCActionSpace(mode="factorized", preset="default")
    return ARCEnv(
        CHALLENGES_JSON,
        SOLUTIONS_JSON,
        action_space=space,
        seed=seed,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Which solvers to test (default: all).",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Show before/after/target plots.",
    )
    p.add_argument(
        "--pair", 
        type=int, 
        help="Only evaluate this 0-based pair index."
        )
    p.add_argument(
        "--stepwise", 
        action="store_true",
        help="Call env.render after every step (requires --pair)."
        )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for determinism.",
    )

    args = p.parse_args()
    if args.stepwise and args.pair is None:
        sys.exit("ERROR: --stepwise requires --pair.")

    keys = args.keys or list_keys()
    missing = [k for k in keys if k not in SOLVERS]
    if missing:
        sys.exit(f"ERROR: no solver stored for {', '.join(missing)}")

    overall_ok = True
    for key in keys:
        env = build_env(seed=args.seed)
        actions = get_actions(key, env.action_space)
        exclude = get_exclude(key)
        results = env.evaluate_sequence(
                        key,
                        actions,
                        render=args.render and not args.stepwise,   # composite fig only if wanted
                        exclude=exclude,
                        include=None if args.pair is None else [args.pair],
                        stepwise=args.stepwise,
                )   

        solved = sum(r["solved"] for r in results)
        total  = len(results)
        ok     = solved == total
        overall_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"{key}: {solved}/{total} solved — {status}")

    if overall_ok:
        print("\n✓ All solvers passed!")
        sys.exit(0)
    else:
        print("\n✗ Some solvers failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
