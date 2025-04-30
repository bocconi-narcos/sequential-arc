# ARC with Reinforcement Learning and Sequential Transformations

This project aims to solve tasks from the Abstraction and Reasoning Corpus (ARC) using Reinforcement Learning (RL) by applying sequential transformations.

## Project Structure

-   **RL Environment:** The core reinforcement learning environment is set up to interact with ARC tasks. It utilizes a specific action space for decision-making.
-   **Action Space (`action_space.py`):** Defines the set of possible actions the RL agent can take.
    -   The specific actions available are configured through `action_config`.
    -   Each action is a composite of three sub-actions:
        1.  **Colour Selection:** Choosing a colour (logic likely in `colour_selection.py` or similar).
        2.  **Selection:** Selecting pixels/objects based on the chosen colour (logic likely in `selection.py` or similar).
        3.  **Transformation:** Applying a transformation to the selected elements (logic likely in `transformation.py` or similar).
-   **Solvers (`solvers/`):** This directory contains specific solvers or approaches developed for individual ARC tasks, potentially serving as baselines or components.

## Current Status

The basic RL environment is implemented, capable of selecting actions defined in `action_space.py` based on the configuration in `action_config`. The composite action structure (colour -> selection -> transformation) is established. The `solvers` directory contains initial solutions for some tasks.