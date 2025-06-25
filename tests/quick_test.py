"""
quick_test.py – run the env + random agent and visualise each move
"""

import pathlib, json, numpy as np
from action_space import ARCActionSpace
from env          import ARCEnv

# ------------------------------------------------------------------ #
#  Dummy data – create tiny JSON files so the env can run standalone
# ------------------------------------------------------------------ #
dummy_challenge = {
    "dummy_key": {
        "train": [
            {"input": [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]],
             "output": [[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]}
        ]
    }
}
dummy_solution = {"dummy_key": 0}

path_c = 'data/challenges.json'
path_s = 'data/solutions.json'

# ------------------------------------------------------------------ #
#  Build action space + environment
# ------------------------------------------------------------------ #
aspace = ARCActionSpace(preset="test", mode="factorized")
env    = ARCEnv(path_c, path_s, action_space=aspace, seed=None)

for j in range(20):
    obs, _ = env.reset(options = {'min_examples':2})
    for step in range(5):
        action = aspace.sample()          # random policy
        obs, r, term, trunc, info = env.step(action)
        print(f"step {step} | reward {r:.1f} | done {term} | trunc {trunc}")
        env.render()
        if term or trunc:
            break
