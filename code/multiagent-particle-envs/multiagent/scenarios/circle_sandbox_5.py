import numpy as np
import math
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 5 with logarithmic rewards and velocity/goal observations.

    Compared to experiment 3 (try not to estimate the target position in the future, like in experiment 4),
    each agent has a dedicated goal to reach.
    - agent 0 is still the leader
    - agent 1 follows the leader (agent_0) or goal_0
    - agent 2 follows agent 1 or goal_1

    Training for 25000 episodes instead of 5000.
    """

    def __init__(self):
        super().__init__()
        self.nb_agents = 3  # including the leader
        self.nb_goals = 2
        self.nb_obstacles = 0

    def reward(self, agent, world):
        """Computes the reward for a given agent in the world using logarithmic distance penalty."""

        # follows the previous agent or its goal
        target_id = agent.id - 1

        # if the goal is activated, try to get it
        landmark = self.find_entity_by_name(world, f"goal_{target_id}")
        if landmark and landmark.activate:
            d = self.dist(agent.state.p_pos, landmark.state.p_pos)

        # else follow the previous agent
        else:
            target = self.find_agent_by_id(world, target_id)
            target_pos = self.estimate_target_pos(agent, target)
            d = self.dist(agent.state.p_pos, target_pos)

        # penalize distance from target
        reward = -math.log(d)
        return reward

    def observation(self, agent, world):
        """Builds local 9-dimensional observation representation for an agent."""
        tg_dx, tg_dy = 0, 0
        tg_vx, tg_vy = 0, 0
        lm_dx, lm_dy, lm_act = 0, 0, 0

        # current agent velocity
        vx = agent.state.p_vel[0]
        vy = agent.state.p_vel[1]

        # follow previous agent or its goal
        target_id = agent.id - 1

        # target distance
        target = self.find_agent_by_id(world, target_id)
        if target:
            self.fix_agent_vel(target)
            tg_dx = target.state.p_pos[0] - agent.state.p_pos[0]
            tg_dy = target.state.p_pos[1] - agent.state.p_pos[1]
            tg_vx = target.state.p_vel[0]
            tg_vy = target.state.p_vel[1]

        # active landmark
        lm = self.find_entity_by_name(world, f"goal_{target_id}")
        if lm:
            lm_dx = lm.state.p_pos[0] - agent.state.p_pos[0]
            lm_dy = lm.state.p_pos[1] - agent.state.p_pos[1]
            lm_act = int(lm.activate)

        # return the complete state
        return np.array([vx, vy, tg_dx, tg_dy, tg_vx, tg_vy, lm_dx, lm_dy, lm_act])
