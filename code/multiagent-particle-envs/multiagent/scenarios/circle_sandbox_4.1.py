import numpy as np
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 4.1, following estimated target positions with bounded linear reward.

    Compared to experiment 4, the reward has been modified to be a bounded linear function of the distance to the goal.
    """

    def __init__(self):
        super().__init__()
        self.nb_agents = 2  # including the leader
        self.nb_goals = 1
        self.nb_obstacles = 0

    def reward(self, agent, world):
        """Computes positive reward based on distance to goal if active, otherwise distance to estimated leader position."""
        leader = world.agents[0]

        # if the goal is activated, try to get it
        lm0 = self.find_entity_by_name(world, "goal_0")
        if lm0 and lm0.activate:
            a_dist = self.dist(agent.state.p_pos, lm0.state.p_pos)
        else:
            # else follow the leader at an estimated lag position
            leader_pos = self.estimate_target_pos_2(agent, leader, coef=4.0)
            a_dist = self.dist(agent.state.p_pos, leader_pos)

        # compared to scenario 4 the reward is not -math.log(a_dist)
        reward = max(1 - a_dist * 0.5, 0.0001)
        return reward

    def observation(self, agent, world):
        """Builds observation incorporating agent velocity, leader distance, and goal state."""
        lm0_dx, lm0_dy, lm0_act = 0, 0, 0
        # current agent velocity
        vx = agent.state.p_vel[0]
        vy = agent.state.p_vel[1]

        # leader distance
        leader = world.agents[0]
        dx = leader.state.p_pos[0] - agent.state.p_pos[0]
        dy = leader.state.p_pos[1] - agent.state.p_pos[1]

        # agent's goal
        lm0 = self.find_entity_by_name(world, "goal_0")
        if lm0:
            lm0_dx = lm0.state.p_pos[0] - agent.state.p_pos[0]
            lm0_dy = lm0.state.p_pos[1] - agent.state.p_pos[1]
            lm0_act = int(lm0.activate)

        # return the complete state
        return np.array([vx, vy, dx, dy, lm0_dx, lm0_dy, lm0_act])
