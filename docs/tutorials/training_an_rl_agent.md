# Training an RL Agent

This tutorial will guide you through the process of training a Reinforcement Learning (RL) agent to solve ARC tasks.

## 1. Set up the Environment

First, you need to create an `ARCEnv` instance. This requires an `ARCActionSpace` instance, which defines the set of possible actions the agent can take.

```python
from action_space import ARCActionSpace
from env import ARCEnv

# Create an action space
aspace = ARCActionSpace(preset="default", mode="factorized")

# Create the environment
env = ARCEnv(
    challenges_json="data/challenges.json",
    solutions_json="data/solutions.json",
    action_space=aspace,
    seed=42
)
```

## 2. Train the Agent

Once you have created the environment, you can train your RL agent. The following code shows a simple example of a random agent.

```python
# Run one episode
obs, _ = env.reset()
terminated = False
truncated = False

while not terminated and not truncated:
    # Sample a random action
    action = env.action_space.sample()

    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Render the environment
    env.render()
```

## 3. Next Steps

This is just a simple example to get you started. To train a more sophisticated RL agent, you can use a library like [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/).
