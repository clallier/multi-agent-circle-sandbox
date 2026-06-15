# Multi-Agent Reinforcement Learning Experiment (Carousel v2)

Welcome to the **multi-agent-maddpg-sandbox** project. This repository contains an experiment focused on Multi-Agent Reinforcement Learning (MARL). Specifically, it utilizes the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train multiple agents in a custom particle-based sandbox environment.

## Overview

This project aims to demonstrate and evaluate the coordination, cooperation, or competition between multiple agents using MARL techniques. It is built upon the foundational concepts from the paper: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

## Project Structure

The project code is divided into two main sub-directories:

*   **`code/maddpg/`**: Contains the core implementation of the MADDPG algorithm, replay buffers, and the training scripts (e.g., `experiments/train.py`).
*   **`code/multiagent-particle-envs/`**: Contains the Multi-Agent Particle Environments (MPE). This includes the physics engine, rendering, and definitions of various scenarios. Custom scenarios (like `circle_sandbox`) are defined in `code/multiagent-particle-envs/multiagent/scenarios/`.

## Prerequisites

Before starting, ensure you have the following installed:

*   [uv](https://github.com/astral-sh/uv) (for environment and dependency management)

## Installation & Setup

Follow these steps to set up the project:

1.  **Clone the repository** and navigate to the project root directory.
2.  **Sync the environment** using `uv`. This will automatically download the correct Python version (3.9) and install all dependencies (including native macOS Apple Silicon compatibility for TensorFlow and setting up the subprojects in editable mode):

    ```bash
    uv sync
    ```

## How It Works

The environment consists of a continuous observation and discrete action space where agents interact based on basic simulated physics.
1.  **Environment Generation:** The `multiagent-particle-envs` directory defines the world, including agents, landmarks, and rules of the specific scenario.
2.  **Training:** The `train.py` script initializes the environment and the MADDPG trainer. During training, agents explore the environment, collect experiences, and update their policies to maximize their rewards over time.
3.  **Evaluation:** Once trained, the saved models can be loaded to visually evaluate the agents' learned behaviors without further training.

## Usage

### 1. Training

To start training the agents (e.g. on scenario 9), run the following command. The models and training states will be saved in the `./test_circle_sandbox_9/` directory.

> [!NOTE]
> We set the environment variable `KERAS_HOME=./.keras` to redirect cache files to a writable workspace directory to avoid sandbox/permission warnings.

```bash
KERAS_HOME=./.keras uv run python code/maddpg/experiments/train.py --scenario circle_sandbox_9 --max-episode-len 80 --num-episodes 5000 --save-rate 200 --save-dir ./test_circle_sandbox_9/
```

### 2. Evaluation

To evaluate the trained agents and display their behavior visually, use the `--display` flag and point to the directory where the model was saved:

```bash
KERAS_HOME=./.keras uv run python code/maddpg/experiments/train.py --scenario circle_sandbox_9 --load-dir ./test_circle_sandbox_9/ --display
```

### 3. Monitoring Training with TensorBoard

You can monitor the training progress, including rewards and losses, using TensorBoard. Run the following command:

```bash
KERAS_HOME=./.keras uv run tensorboard --logdir ./test_circle_sandbox_9/
```
*(Once running, open the provided localhost link in your web browser to view the dashboard).*

### 4. Additional Information

For a full list of command-line options available for training and environment configuration, you can use the help flag:

```bash
uv run python code/maddpg/experiments/train.py --help
```

*Note: New scenario scripts should be defined in the `code/multiagent-particle-envs/multiagent/scenarios/` directory.*
