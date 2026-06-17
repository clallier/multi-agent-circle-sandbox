from random import randint, randrange
import numpy as np
import math
from multiagent.core import World, Agent, Landmark, Action
from multiagent.scenario import BaseScenario


class CircleSandboxBaseScenario(BaseScenario):
    """Base Scenario for Multi-Agent Circle Sandbox experiments, containing common utilities.

    This class centralizes lookup utilities, mathematical operations, target position estimation,
    standard environment world creation/reset, and leader trajectory callbacks.
    """

    def __init__(self):
        super().__init__()
        self.nb_agents = 3
        self.nb_goals = 2
        self.nb_obstacles = 0
        self.leader_callback_type = "circle"

    def create_leader_callback(self):
        """Generates action callback for circular trajectory of the leader."""
        if self.leader_callback_type == "modulated":
            return self.get_modulated_leader_callback()
        return self.get_circle_leader_callback()

    def dist(self, a, b):
        """Calculates the Euclidean distance between two 2D points.

        Args:
            a (np.ndarray): First point.
            b (np.ndarray): Second point.

        Returns:
            float: Euclidean distance.
        """
        return self._quick_norm(a - b)

    def _quick_norm(self, vec):
        """Compute L2 norm of a 2D vector using math.hypot for better precision.

        Args:
            vec (np.ndarray): 2D vector.

        Returns:
            float: L2 norm of the vector.
        """
        return math.hypot(vec[0], vec[1])

    def make_world(self):
        """Creates and configures the environment world using subclass settings.

        Returns:
            World: The configured simulation world.
        """
        world = World()

        # add agents
        world.agents = [Agent() for _ in range(self.nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.id = i
            agent.collide = True
            agent.color = [1, 1, 0, 0.75]
            agent.silent = True
            agent.size = 0.1

        # add goals
        goals = [Landmark() for _ in range(self.nb_goals)]
        for i, goal in enumerate(goals):
            goal.name = "goal_%d" % i
            goal.type = "goal"
            goal.color = [1, 0, 0, 0.75]
            goal.collide = False
            goal.movable = False
            goal.size = 0.2

        # add obstacles
        obstacles = [Landmark() for _ in range(self.nb_obstacles)]
        for i, obstacle in enumerate(obstacles):
            obstacle.name = "obstacle_%d" % i
            obstacle.type = "obstacle"
            obstacle.collide = True
            obstacle.movable = False
            obstacle.color = [1, 1, 1, 0.75]
            obstacle.size = 0.1

        world.landmarks = goals + obstacles

        # Cache entities by name and agents by id for O(1) lookups
        # pyrefly: ignore [missing-attribute]
        world._entities_by_name = {e.name: e for e in world.entities}
        # pyrefly: ignore [missing-attribute]
        world._agents_by_id = {a.id: a for a in world.agents}

        # make initial conditions
        self.reset_world(world)

        return world

    def configure_goal_timing(self, goal, i, total_goals):
        """Configures the default activation, deactivation, and count tracking for a goal.

        This centralizes the timing logic that controls when landmarks switch
        activation states, preventing inconsistencies and allowing subclasses to customize.

        Args:
            goal (Landmark): The goal landmark to configure.
            i (int): The 0-based index of the goal.
            total_goals (int): Total number of goals in the environment.

        Returns:
            None: Modifies the goal landmark object in place.

        Raises:
            ValueError: If the goal parameters are invalid.

        Examples:
            >>> scenario.configure_goal_timing(goal, 0, 3)
        """
        base_activation_time = 20
        goal.next_activate_time_min = 15
        goal.next_activate_time_max = 25
        goal.activate_time = i * base_activation_time + randrange(
            goal.next_activate_time_min, goal.next_activate_time_max
        )
        goal.deactivation_time = (1 + total_goals) * base_activation_time
        goal.activate_count = 1

    def reset_world(self, world):
        """Resets the state of the world and positions all entities.

        Args:
            world (World): The environment world.
        """
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0, 0, 1, 1])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.prev_state = agent.state

            if i == 0:
                # the first agent is the leader to be followed
                agent.collide = False
                agent.action_callback = self.create_leader_callback()
                agent.action_callback(agent, world)
                agent.size = 0.1
                agent.color = [0, 0, 1, 0.75]
            else:
                # others are learning agents
                agent.size = 0.05
                agent.color = [0.2 * i, 0, 1, 0.75]

        goals = [ld for ld in world.landmarks if ld.type == "goal"]
        obstacles = [ld for ld in world.landmarks if ld.type == "obstacle"]

        for i, goal in enumerate(goals):
            goal.color = np.array([1, 1, 0, 0.75])
            goal.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            goal.state.p_vel = np.zeros(world.dim_p)
            goal.color = np.array([0, 1, 0, 0.75])
            goal.activate = False
            # default activate_time setting, can be disabled/overridden in subclass
            self.configure_goal_timing(goal, i, len(goals))

        for obstacle in obstacles:
            obstacle.color = np.array([1, 1, 1, 0.75])
            obstacle.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)

    def find_entity_by_name(self, world, name: str):
        """Finds a landmark entity by its name.

        Args:
            world (World): The environment world.
            name (str): Entity name.

        Returns:
            Landmark: Landmark entity if found.
        """
        if not hasattr(world, "_entities_by_name"):
            world._entities_by_name = {e.name: e for e in world.entities}
        return world._entities_by_name.get(name, None)

    def find_agent_by_id(self, world, id: int):
        """Finds an agent by its ID.

        Args:
            world (World): The environment world.
            id (int): Agent ID.

        Returns:
            Agent: Agent entity if found.
        """
        if not hasattr(world, "_agents_by_id"):
            world._agents_by_id = {a.id: a for a in world.agents}
        return world._agents_by_id.get(id, None)

    def get_last_active_landmark(self, world):
        """Finds the first active goal landmark in the world.

        Args:
            world (World): The environment world.

        Returns:
            Landmark: The first active goal landmark if found, otherwise None.
        """
        last_landmark = None
        for landmark in world.landmarks:
            if landmark.type == "goal" and landmark.activate:
                last_landmark = landmark
        return last_landmark

    def estimate_target_pos(self, agent, target):
        """Estimates target position. Defaults to estimator 1.

        Args:
            agent (Agent): Tracking agent.
            target (Agent): Target agent to track.

        Returns:
            np.ndarray: Target position estimate.
        """
        return self.estimate_target_pos_1(agent, target)

    def estimate_target_pos_1(self, agent, target, coef=2.0):
        """Estimates target position using target's current position directly.

        This is the simplest baseline estimator returning target's exact position.

        Args:
            agent (Agent): Tracking agent.
            target (Agent): Target agent to track.
            coef (float, optional): Unused multiplier coefficient. Defaults to 2.0.

        Returns:
            np.ndarray: Estimated target position (same as target.state.p_pos).
        """
        return target.state.p_pos

    def estimate_target_pos_2(self, agent, target, coef=4.0):
        """Estimates target position by projecting backwards from target's previous position.

        Uses the target's previous velocity vector scaled by coef to estimate a lagged position.

        Args:
            agent (Agent): Tracking agent.
            target (Agent): Target agent to track.
            coef (float, optional): Multiplier coefficient for velocity vector. Defaults to 4.0.

        Returns:
            np.ndarray: Estimated target position.
        """
        return target.prev_state.p_pos - (target.prev_state.p_vel * coef)

    def estimate_target_pos_3(self, agent, target, coef=2.0):
        """Estimates target position by projecting backwards from target's current position.

        Uses the target's current velocity vector scaled by coef to estimate a lagged position.

        Args:
            agent (Agent): Tracking agent.
            target (Agent): Target agent to track.
            coef (float, optional): Multiplier coefficient for velocity vector. Defaults to 2.0.

        Returns:
            np.ndarray: Estimated target position.
        """
        return target.state.p_pos - (target.state.p_vel * coef)

    def estimate_target_pos_4(self, agent, target, coef=0.1):
        """Estimates target position offset along the line from tracking agent to target.

        Pulls the estimated position slightly towards the tracking agent by distance coef.

        Args:
            agent (Agent): Tracking agent.
            target (Agent): Target agent to track.
            coef (float, optional): Distance to shift estimated position. Defaults to 0.1.

        Returns:
            np.ndarray: Estimated target position.
        """
        delta = target.state.p_pos - agent.state.p_pos
        mag = np.linalg.norm(delta)
        new_target = (
            target.state.p_pos - (coef * delta / mag) if mag > 0 else target.state.p_pos
        )
        return new_target

    def estimate_target_pos_5(self, agent, target, coef=0.2):
        """Estimates target position by offsetting backwards along target's velocity unit vector.

        Shifts the position in the opposite direction of the target's current movement.

        Args:
            agent (Agent): Tracking agent.
            target (Agent): Target agent to track.
            coef (float, optional): Distance coefficient to shift backwards. Defaults to 0.2.

        Returns:
            np.ndarray: Estimated target position.
        """
        p_vel_mag = np.linalg.norm(target.state.p_vel)
        p_vel_u = target.state.p_vel / p_vel_mag if p_vel_mag > 0 else np.array([0, 0])
        new_vel = -coef * p_vel_u
        new_pos = target.state.p_pos + new_vel
        return new_pos

    def get_angle_unsigned(self, v1, v2):
        """Computes the unsigned angle between two vectors using arccos.

        Args:
            v1 (np.ndarray): First vector.
            v2 (np.ndarray): Second vector.

        Returns:
            float: Unsigned angle in radians.
        """
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        v1_u = v1 / v1_norm if v1_norm > 0 else np.array([0, 0])
        v2_u = v2 / v2_norm if v2_norm > 0 else np.array([0, 0])
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def get_angle_signed_and_mag(self, v1, v2):
        """Computes the signed angle and magnitude difference using arctan2.

        Args:
            v1 (np.ndarray): First vector.
            v2 (np.ndarray): Second vector.

        Returns:
            tuple: (signed angle, magnitude difference).
        """
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        v1_u = v1 / v1_mag if v1_mag > 0 else np.array([0, 0])
        v2_u = v2 / v2_mag if v2_mag > 0 else np.array([0, 0])

        dv_dot = np.dot(v1_u, v2_u)
        dv_dot = 0.0 if np.isnan(dv_dot) else dv_dot
        dv_cross = np.cross(v1_u, v2_u)
        dv_cross = 0.0 if np.isnan(dv_cross) else dv_cross

        dv_mag = abs(v1_mag - v2_mag)
        dv_angle = np.arctan2(dv_cross, dv_dot)
        return dv_angle, dv_mag

    def cos_sim(self, v1, v2):
        """Computes cosine similarity between two vectors.

        Args:
            v1 (np.ndarray): First vector.
            v2 (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity value.
        """
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        return np.dot(v1, v2) / (v1_norm * v2_norm)

    def sigmoid(self, x):
        """Computes a custom sigmoid function.

        Args:
            x (float): Input value.

        Returns:
            float: Sigmoid mapped value.
        """
        var = 4
        shift = 9
        return 1 / (1 + np.exp(-var + (x * shift)))

    def fix_agent_vel(self, agent):
        """Fixes zero velocities to small values to avoid division by zero.

        Args:
            agent (Agent): Agent to update.
        """
        if (
            agent.state.p_vel[0] == 0
            and agent.state.p_vel[1] == 0
            and hasattr(agent.state, "speed")
        ):
            agent.state.p_vel = agent.state.speed
        if agent.state.p_vel[0] == 0:
            agent.state.p_vel[0] = 0.00001
        if agent.state.p_vel[1] == 0:
            agent.state.p_vel[1] = 0.00001

    def _compute_next_position(self, phase, scale=2 / 3):
        """Computes the 2D coordinate on a circle from a given phase.

        Args:
            phase (float): The current phase step.
            scale (float): Circle radius scale. Defaults to 2/3.

        Returns:
            np.ndarray: 2D coordinate vector.
        """
        scaled_step = (phase / 50) * (2 * math.pi) * 0.75
        return np.array([math.sin(scaled_step), math.cos(scaled_step)]) * scale

    def _get_leader_callback_helper(self, modulated=False):
        """Generates a trajectory callback for the leader.

        Args:
            modulated (bool): If True, modulates the speed dynamically. Defaults to False.

        Returns:
            callable: Action callback function.
        """
        phase = float(randint(1, 100))
        direction = 1.0 if randint(0, 1) == 0 else -1.0
        last_pos = None

        def cb(agent, world):
            nonlocal phase, last_pos
            act = Action()
            rng = randint(20, 70) if modulated else 50.0
            phase += direction * (50.0 / rng)

            new_pos = self._compute_next_position(phase)
            agent.state.p_pos = new_pos

            if last_pos is not None:
                agent.state.speed = new_pos - last_pos
            else:
                agent.state.speed = np.array([0.0, 0.0])

            last_pos = new_pos
            act.u = np.zeros(2)
            return act

        return cb

    def get_modulated_leader_callback(self):
        """Generates the modulated leader trajectory callback with continuous phase noise.

        Returns:
            callable: Action callback function of signature `cb(agent, world) -> Action`.
        """
        return self._get_leader_callback_helper(modulated=True)

    def get_circle_leader_callback(self):
        """Generates the standard circular leader trajectory callback.

        Returns:
            callable: Action callback function of signature `cb(agent, world) -> Action`.
        """
        return self._get_leader_callback_helper(modulated=False)
