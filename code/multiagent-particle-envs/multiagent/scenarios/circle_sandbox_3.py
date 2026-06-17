import numpy as np
import math
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 3, incorporating a single goal that can activate.

    Compared to experiment 2, the reward takes into account the current agent's target (leader or activated goal).
    """

    def __init__(self):
        super().__init__()
        self.nb_agents = 2  # including the leader
        self.nb_goals = 1
        self.nb_obstacles = 0

    def reward(self, agent, world):
        """Computes reward based on distance to goal if active, otherwise distance to leader."""
        # if the goal is activated, try to get it
        lm0 = self.find_entity_by_name(world, "goal_0")
        if lm0 and lm0.activate:
            a_dist = self.dist(agent.state.p_pos, lm0.state.p_pos)
        else:
            # else follow the leader
            a_dist = self.dist(agent.state.p_pos, world.agents[0].state.p_pos)

        reward = -math.log(a_dist)
        return reward

    def observation(self, agent, world):
        """Builds observation incorporating agent velocity, leader distance, and goal state."""
        lm0_dx, lm0_dy, lm0_act = 0, 0, 0
        # current agent velocity
        vx = agent.state.p_vel[0]
        vy = agent.state.p_vel[1]

        # leader distance
        leader = world.agents[0]
        dx = leader.state.p_pos[0] + leader.state.speed[0] - agent.state.p_pos[0]
        dy = leader.state.p_pos[1] + leader.state.speed[1] - agent.state.p_pos[1]

        # agent's goal
        lm0 = self.find_entity_by_name(world, "goal_0")
        if lm0:
            lm0_dx = lm0.state.p_pos[0] - agent.state.p_pos[0]
            lm0_dy = lm0.state.p_pos[1] - agent.state.p_pos[1]
            lm0_act = int(lm0.activate)

        # return the complete state
        return np.array([vx, vy, dx, dy, lm0_dx, lm0_dy, lm0_act])
