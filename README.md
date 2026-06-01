# Multi-Agent Reinforcement Learning Experiment (Carousel v2)

Welcome to the **technical-exercice-carousel-v2** project. This repository contains an experiment focused on Multi-Agent Reinforcement Learning (MARL). Specifically, it utilizes the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train multiple agents in a custom particle-based sandbox environment.

## Overview

This project aims to demonstrate and evaluate the coordination, cooperation, or competition between multiple agents using MARL techniques. It is built upon the foundational concepts from the paper: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

## Project Structure

The project code is divided into two main sub-directories:

*   **`code/maddpg/`**: Contains the core implementation of the MADDPG algorithm, replay buffers, and the training scripts (e.g., `experiments/train.py`).
*   **`code/multiagent-particle-envs/`**: Contains the Multi-Agent Particle Environments (MPE). This includes the physics engine, rendering, and definitions of various scenarios. Custom scenarios (like `circle_sandbox`) are defined in `code/multiagent-particle-envs/multiagent/scenarios/`.

## Prerequisites

Before starting, ensure you have the following installed:

*   Python (3.5+)
*   `pip` (Python package installer)

## Installation

Follow these steps to set up the project:

1.  **Clone the repository** (if you haven't already) and navigate to the project root directory.
2.  **Install the dependencies** using the provided `requirements.txt` file. This will install required packages like TensorFlow, NumPy, Pyglet, and also install the local `maddpg` and `multiagent-particle-envs` packages in editable mode.

    ```bash
    pip install -r requirements.txt
    ```

## How It Works

The environment consists of a continuous observation and discrete action space where agents interact based on basic simulated physics.
1.  **Environment Generation:** The `multiagent-particle-envs` directory defines the world, including agents, landmarks, and rules of the specific scenario.
2.  **Training:** The `train.py` script initializes the environment and the MADDPG trainer. During training, agents explore the environment, collect experiences, and update their policies to maximize their rewards over time.
3.  **Evaluation:** Once trained, the saved models can be loaded to visually evaluate the agents' learned behaviors without further training.

## Usage

### 1. Training

To start training the agents on the `circle_sandbox` scenario, run the following command. The models and training states will be saved in the `./test_circle_sandbox/` directory.

```bash
python code/maddpg/experiments/train.py --scenario circle_sandbox --max-episode-len 80 --num-episodes 5000 --save-rate 200 --save-dir ./test_circle_sandbox/
```

### 2. Evaluation

To evaluate the trained agents and display their behavior visually, use the `--display` flag and point to the directory where the model was saved:

```bash
python code/maddpg/experiments/train.py --scenario circle_sandbox --load-dir ./test_circle_sandbox/ --display
```

### 3. Monitoring Training with TensorBoard

You can monitor the training progress, including rewards and losses, using TensorBoard. Run the following command, pointing `--logdir` to your save directory:

```bash
tensorboard --logdir=./test_circle_sandbox
```
*(Once running, open the provided localhost link in your web browser to view the dashboard).*

### 4. Additional Information

For a full list of command-line options available for training and environment configuration, you can use the help flag:

```bash
python code/maddpg/experiments/train.py --help
```

*Note: New scenario scripts should be defined in the `code/multiagent-particle-envs/multiagent/scenarios/` directory.*
