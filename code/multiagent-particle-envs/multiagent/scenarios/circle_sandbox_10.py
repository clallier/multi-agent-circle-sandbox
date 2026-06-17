import numpy as np
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 10 with bounded exponential decay rewards and domain randomization.

    Compared to scenario 9, the agent states take into account other agents pos/vel to avoid collision.

    Training for 100_000 episodes
    """

    def __init__(self):
        super().__init__()
        self.nb_agents = 4  # including the leader
        self.nb_goals = 3
        self.nb_obstacles = 0
        self.leader_callback_type = "modulated"

    def reward(self, agent, world):
        """Computes the overall reward for a given agent including obstacle penalties.

        The reward consists of a target-reaching or leader-following component
        using Bounded Exponential Decay, combined with a proximity penalty for any
        obstacles in the environment to prevent collisions.

        Args:
            agent (Agent): The agent entity for whom the reward is computed.
            world (World): The environment world instance.

        Returns:
            float: The computed reward value.

        Raises:
            ValueError: If target or landmark lookup fails.

        Examples:
            >>> r = self.reward(agent, world)
        """
        sigma_d = 1.0
        base_reward = 0.0
        landmark = self.get_last_active_landmark(world)
        if landmark:
            d = self.dist(agent.state.p_pos, landmark.state.p_pos)
            base_reward = np.exp(-d / sigma_d)
        else:
            target_id = agent.id - 1
            target = self.find_agent_by_id(world, target_id)
            target_pos = self.estimate_target_pos(agent, target)
            d = self.dist(agent.state.p_pos, target_pos)
            r_dist = np.exp(-d / sigma_d)
            cos_sim = max(0, self.cos_sim(target.state.p_vel, agent.state.p_vel))
            base_reward = 0.7 * r_dist + 0.3 * cos_sim
        return base_reward

    def observation(self, agent, world):
        """Builds local observation representation for an agent."""
        d_pos, lm_d_pos = (
            np.array([0, 0]),
            np.array([0, 0]),
        )
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
            # delta vel (signed angle and magnitude)
            dv_angle, dv_mag = self.get_angle_signed_and_mag(
                target.state.p_vel, agent.state.p_vel
            )

        # active landmark
        lm = self.get_last_active_landmark(world)
        if lm:
            lm_d_pos = lm.state.p_pos - agent.state.p_pos
            lm_act = int(lm.activate)

        # relative positions of other agents (excluding leader and target agent)
        other_d_pos = []
        other_d_vel = []
        for other in world.agents:
            if other.id == 0 or other.id == agent.id or other.id == target_id:
                continue
            other_d_pos.append(other.state.p_pos - agent.state.p_pos)
            other_d_vel.append(
                self.get_angle_signed_and_mag(other.state.p_vel, agent.state.p_vel)
            )

        # return the complete state
        base_obs = [
            vx,
            vy,
            d_pos[0],
            d_pos[1],
            dv_angle,
            dv_mag,
            lm_d_pos[0],
            lm_d_pos[1],
            lm_act,
        ]
        for dp in other_d_pos:
            base_obs.append(dp[0])
            base_obs.append(dp[1])
        for dv in other_d_vel:
            base_obs.append(dv[0])
            base_obs.append(dv[1])

        return np.array(base_obs)
