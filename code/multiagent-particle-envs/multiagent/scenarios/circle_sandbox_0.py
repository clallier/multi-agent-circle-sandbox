import numpy as np
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 0 with dummy rewards and observations."""

    def __init__(self):
        super().__init__()
        self.nb_agents = 2  # including the leader
        self.nb_goals = 0
        self.nb_obstacles = 0

    def reward(self, agent, world):
        """Computes dummy reward."""
        return 0.0

    def observation(self, agent, world):
        """Builds dummy observation."""
        return np.array([0.0])

