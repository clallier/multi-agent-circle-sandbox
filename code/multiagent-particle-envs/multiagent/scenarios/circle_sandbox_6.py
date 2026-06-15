import numpy as np
import math
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 6 with logarithmic rewards and velocity/goal observations.

    Inherits helper routines, world setup, and callback factories from CircleSandboxBaseScenario.
    """

    def __init__(self):
        super().__init__()
        self.nb_agents = 3  # including the leader
        self.nb_goals = 2
        self.nb_obstacles = 0

    def get_angle(self, v1, v2):
        return self.get_angle_unsigned(v1, v2)

    def reward(self, agent, world):
        """Computes the reward for a given agent in the world using logarithmic distance penalty."""
        target_id = agent.id - 1
        target = self.find_agent_by_id(world, target_id)

        # if the goal is activated, try to get it
        landmark = self.find_entity_by_name(world, f"Goal {target_id}")
        if landmark and landmark.activate:
            d = self.dist(agent.state.p_pos, landmark.state.p_pos)
            reward = -math.log(d)
        # else follow the leader
        else:
            target_pos = self.estimate_target_pos(agent, target)
            d = self.dist(agent.state.p_pos, target_pos)
            d_norm = -math.log(d)
            cos_sim = max(0, self.cos_sim(target.state.p_vel, agent.state.p_vel))
            reward = 0.7 * d_norm + 0.3 * cos_sim
        return reward

    def observation(self, agent, world):
        """Builds local 9-dimensional observation representation for an agent."""
        tg_dx, tg_dy = 0, 0
        tg_vx, tg_vy = 0, 0
        lm_dx, lm_dy, lm_act = 0, 0, 0

        # current agent velocity
        vx = agent.state.p_vel[0]
        vy = agent.state.p_vel[1]

        target_id = agent.id - 1

        # target distance
        target = self.find_agent_by_id(world, target_id)
        if target:
            self.fix_agent_vel(target)
            tg_dx = target.state.p_pos[0] - agent.state.p_pos[0]
            tg_dy = target.state.p_pos[1] - agent.state.p_pos[1]
            tg_vx = target.state.p_vel[0]
            tg_vy = target.state.p_vel[1]

        # agent's goal
        lm = self.find_entity_by_name(world, f"Goal {target_id}")
        if lm:
            lm_dx = lm.state.p_pos[0] - agent.state.p_pos[0]
            lm_dy = lm.state.p_pos[1] - agent.state.p_pos[1]
            lm_act = int(lm.activate)

        # return the complete state
        return np.array([vx, vy, tg_dx, tg_dy, tg_vx, tg_vy, lm_dx, lm_dy, lm_act])

