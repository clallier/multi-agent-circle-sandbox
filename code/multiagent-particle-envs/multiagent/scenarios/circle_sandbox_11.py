from multiagent import palette
import numpy as np
from multiagent.scenarios.circle_sandbox_base import CircleSandboxBaseScenario


class Scenario(CircleSandboxBaseScenario):
    """Scenario representing experiment 11 with boids swarm behaviors.

    This scenario implements alignment, coherence, and separation behaviors
    among agents tracking a leader or an active goal landmark.
    """

    # Internal Constants
    SIGMA_DIST = 1.0
    SIGMA_ALIGN_ANGLE = 0.5
    SIGMA_ALIGN_MAG = 0.5
    SIGMA_COHERE = 1.0
    SIGMA_SEPARATE = 0.1
    NEIGHBOR_RADIUS = 0.2
    SEPARATION_RADIUS = 0.05

    TRACK_WEIGHT = 0.4
    ALIGN_ANGLE_WEIGHT = 0.15
    ALIGN_MAG_WEIGHT = 0.15
    COHERENCE_WEIGHT = 0.3
    SEPARATION_WEIGHT = 1.0

    def __init__(self):
        super().__init__()
        self.nb_agents = 10  # including the leader
        self.nb_goals = 3
        self.nb_obstacles = 0
        self.leader_callback_type = "modulated"

    def reset_world(self, world):
        palette.AGENT_SIZE = 0.025
        super().reset_world(world)

    def _precompute_matrices(self, world):
        """Precomputes and caches distance and velocity relationships.

        How it works:
            Checks if cached keys match current agent states. If not, updates
            the cached state key and recalculates pairwise relations.

        Args:
            world (World): The environment world.

        Returns:
            None
        """
        state_key = tuple(
            (tuple(a.state.p_pos), tuple(a.state.p_vel)) for a in world.agents
        )
        if hasattr(self, "_cached_state_key") and self._cached_state_key == state_key:
            return

        n = len(world.agents)
        self._cached_dist_matrix = np.zeros((n, n))
        self._cached_vel_angle = np.zeros((n, n))
        self._cached_vel_mag = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    self._update_pair_cache(world, i, j)
        self._cached_state_key = state_key

    def _update_pair_cache(self, world, i, j):
        """Computes and caches relationship values for a pair of agents.

        How it works:
            Calculates distance and velocity differences, and stores them in
            their respective matrices.

        Args:
            world (World): The environment world.
            i (int): Index of the first agent.
            j (int): Index of the second agent.

        Returns:
            None
        """
        agent_i = world.agents[i]
        agent_j = world.agents[j]
        self._cached_dist_matrix[agent_i.id, agent_j.id] = self.dist(
            agent_i.state.p_pos, agent_j.state.p_pos
        )
        angle, mag = self.get_angle_signed_and_mag(
            agent_j.state.p_vel, agent_i.state.p_vel
        )
        self._cached_vel_angle[agent_i.id, agent_j.id] = angle
        self._cached_vel_mag[agent_i.id, agent_j.id] = mag

    def _compute_neighbor_averages(self, agent, world):
        """Calculates average values for neighbors within the radius.

        How it works:
            Finds all other agents within the neighborhood radius, accumulates
            their position offsets and velocity parameters, and returns their averages.

        Args:
            agent (Agent): The current agent.
            world (World): The environment world.

        Returns:
            tuple: (avg_delta_pos, avg_angle, avg_mag)
        """
        self._precompute_matrices(world)
        d_pos, angles, mags = [], [], []
        for other in world.agents:
            if other.id == agent.id:
                continue
            dist = self._cached_dist_matrix[agent.id, other.id]
            if dist <= self.NEIGHBOR_RADIUS:
                d_pos.append(other.state.p_pos - agent.state.p_pos)
                angles.append(self._cached_vel_angle[agent.id, other.id])
                mags.append(self._cached_vel_mag[agent.id, other.id])

        if not d_pos:
            return np.zeros(world.dim_p), 0.0, 0.0
        return np.mean(d_pos, axis=0), np.mean(angles), np.mean(mags)

    def observation(self, agent, world):
        """Builds observation representation for scenario 11.

        How it works:
            Gathers velocities, landmark relative position/activation,
            leader relative position/velocity matching parameters, and
            neighbor average position/velocity matching parameters.

        Args:
            agent (Agent): The current agent.
            world (World): The environment world.

        Returns:
            np.ndarray: Built observation vector.
        """
        lm = self.get_last_active_landmark(world)
        lm_d_pos = lm.state.p_pos - agent.state.p_pos if lm else np.zeros(2)
        lm_act = int(lm.activate) if lm else 0

        leader = self.find_agent_by_id(world, 0)
        self.fix_agent_vel(leader)
        leader_d = leader.state.p_pos - agent.state.p_pos
        l_angle, l_mag = self.get_angle_signed_and_mag(
            leader.state.p_vel, agent.state.p_vel
        )

        n_pos, n_angle, n_mag = self._compute_neighbor_averages(agent, world)

        return np.array([
            agent.state.p_vel[0],
            agent.state.p_vel[1],
            lm_d_pos[0],
            lm_d_pos[1],
            lm_act,
            leader_d[0],
            leader_d[1],
            l_angle,
            l_mag,
            n_pos[0],
            n_pos[1],
            n_angle,
            n_mag,
        ])

    def reward(self, agent, world):
        """Computes the overall reward for a given agent in scenario 11.

        How it works:
            Combines tracking reward (target distance) with boids swarm
            rewards (alignment, coherence, and separation).

        Args:
            agent (Agent): The agent entity for whom the reward is computed.
            world (World): The environment world instance.

        Returns:
            float: The computed reward value.
        """
        target_pos = self._get_target_pos(agent, world)
        d = self.dist(agent.state.p_pos, target_pos)
        r_track = np.exp(-d / self.SIGMA_DIST)

        if not self._has_neighbors(agent, world):
            return r_track

        n_pos, n_angle, n_mag = self._compute_neighbor_averages(agent, world)
        r_align_angle = np.exp(-abs(n_angle) / self.SIGMA_ALIGN_ANGLE)
        r_align_mag = np.exp(-abs(n_mag) / self.SIGMA_ALIGN_MAG)
        d_avg = self._quick_norm(n_pos)
        r_cohere = np.exp(-d_avg / self.SIGMA_COHERE)
        # r_separate is a penalty  [0, -1]
        r_separate = (
            np.exp(-d_avg / self.SIGMA_SEPARATE)
            if d_avg < self.SEPARATION_RADIUS
            else 0.0
        )

        return (
            self.TRACK_WEIGHT * r_track
            + self.ALIGN_ANGLE_WEIGHT * r_align_angle
            + self.ALIGN_MAG_WEIGHT * r_align_mag
            + self.COHERENCE_WEIGHT * r_cohere
            - self.SEPARATION_WEIGHT * r_separate
        )

    def _get_target_pos(self, agent, world):
        """Retrieves the position of the current target (landmark or leader).

        How it works:
            Looks for an active landmark goal. If none exists, targets the leader.

        Args:
            agent (Agent): The agent entity.
            world (World): The environment world instance.

        Returns:
            np.ndarray: Position of the target.
        """
        lm = self.get_last_active_landmark(world)
        if lm:
            return lm.state.p_pos
        leader = self.find_agent_by_id(world, 0)
        return self.estimate_target_pos(agent, leader)

    def _has_neighbors(self, agent, world):
        """Checks if the agent has any neighbors within NEIGHBOR_RADIUS.

        How it works:
            Iterates through all agents and returns True if a neighbor is found within range.

        Args:
            agent (Agent): The current agent.
            world (World): The environment world.

        Returns:
            bool: True if there is at least one neighbor.
        """
        for other in world.agents:
            if (
                other.id != agent.id
                and self._cached_dist_matrix[agent.id, other.id] <= self.NEIGHBOR_RADIUS
            ):
                return True
        return False
