import numpy as np
import math
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 2 with distance relative to leader velocity for observation."""

    def __init__(self):
        super().__init__()
        self.nb_agents = 2  # including the leader
        self.nb_goals = 0
        self.nb_obstacles = 0

    def reward(self, agent, world):
        """Computes logarithmic reward based on distance to the leader."""
        a_dist = self.dist(agent.state.p_pos, world.agents[0].state.p_pos)
        reward = -math.log(a_dist)
        return reward

    def observation(self, agent, world):
        """Builds observation containing relative offset (incorporating speed) and agent velocity."""
        leader = world.agents[0]
        dx = leader.state.p_pos[0] + leader.state.speed[0] - agent.state.p_pos[0]
        dy = leader.state.p_pos[1] + leader.state.speed[1] - agent.state.p_pos[1]
        return np.array([dx, dy, agent.state.p_vel[0], agent.state.p_vel[1]])

