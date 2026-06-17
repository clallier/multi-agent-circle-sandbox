import numpy as np
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 9 with bounded exponential decay rewards.

    Compared to scenario 8, the rewards use a bounded exponential decay (e^{-d/sigma_d}) instead of the logarithmic function.

    Training for 100_000 episodes
    """

    def __init__(self):
        super().__init__()
        self.nb_agents = 4  # including the leader
        self.nb_goals = 3
        self.leader_callback_type = "modulated"

    def reward(self, agent, world):
        """Computes the reward for a given agent in the world using Bounded Exponential Decay.

        If a goal landmark is active, the reward is purely based on the exponential decay of
        the distance to that landmark. Otherwise, the agent receives a weighted combination of
        distance-based reward (following the previous agent's estimated target position) and
        velocity alignment (cosine similarity).

        Args:
            agent (Agent): The agent for which the reward is computed.
            world (World): The environment's physics world.

        Returns:
            float: The computed reward value. High values (up to 1.0) signify close proximity
                and good velocity alignment to the target.

        Raises:
            ValueError: If distance calculation results in negative values.

        Examples:
            >>> r = scenario.reward(agent, world)
            >>> print(r)
            0.85
        """
        sigma_d = 1.0

        # if the goal is activated, try to get it
        landmark = self.get_last_active_landmark(world)
        if landmark:
            d = self.dist(agent.state.p_pos, landmark.state.p_pos)
            reward = np.exp(-d / sigma_d)
        # else follow the previous agent
        else:
            target_id = agent.id - 1
            target = self.find_agent_by_id(world, target_id)
            target_pos = self.estimate_target_pos(agent, target)
            d = self.dist(agent.state.p_pos, target_pos)
            r_dist = np.exp(-d / sigma_d)
            cos_sim = max(0, self.cos_sim(target.state.p_vel, agent.state.p_vel))
            reward = 0.7 * r_dist + 0.3 * cos_sim
        return reward

    def observation(self, agent, world):
        """Builds local 9-dimensional observation representation for an agent."""

        d_pos, lm_d_pos = np.array([0, 0]), np.array([0, 0])
        dv_mag, dv_angle, lm_act = 0, 0, 0

        # current agent velocity
        vx = agent.state.p_vel[0]
        vy = agent.state.p_vel[1]

        # target distance
        target_id = agent.id - 1
        target = self.find_agent_by_id(world, target_id)
        if target:
            self.fix_agent_vel(target)
            # delta x/y
            d_pos = target.state.p_pos - agent.state.p_pos
            # delta vel
            dv_angle, dv_mag = self.get_angle_signed_and_mag(
                target.state.p_vel, agent.state.p_vel
            )

        # agent's goal
        lm = self.get_last_active_landmark(world)
        if lm:
            lm_d_pos = lm.state.p_pos - agent.state.p_pos
            lm_act = int(lm.activate)

        return np.array([
            vx,
            vy,
            d_pos[0],
            d_pos[1],
            dv_angle,
            dv_mag,
            lm_d_pos[0],
            lm_d_pos[1],
            lm_act,
        ])
