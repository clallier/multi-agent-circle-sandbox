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
            agent.name = "agent %d" % i
            agent.id = i
            agent.collide = True
            agent.color = [1, 1, 0, 1]
            agent.silent = True
            agent.size = 0.1

        # add goals
        goals = [Landmark() for _ in range(self.nb_goals)]
        for i, goal in enumerate(goals):
            goal.name = "Goal %d" % i
            goal.type = "goal"
            goal.color = [1, 0, 0, 0.75]
            goal.collide = False
            goal.movable = False
            goal.size = 0.2

        # add obstacles
        obstacles = [Landmark() for _ in range(self.nb_obstacles)]
        for i, obstacle in enumerate(obstacles):
            obstacle.name = "Obstacle %d" % i
            obstacle.type = "obstacle"
            obstacle.collide = True
            obstacle.movable = False
            obstacle.color = [1, 0, 1, 1]
            obstacle.size = 0.1

        world.landmarks = goals + obstacles

        # Cache entities by name and agents by id for O(1) lookups
        world._entities_by_name = {e.name: e for e in world.entities}
        world._agents_by_id = {a.id: a for a in world.agents}

        # make initial conditions
        self.reset_world(world)

        return world

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
                agent.color = [0, 0, 1]
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
            goal.activate_time = (i + 1) * 30 + randrange(5, 10)
            goal.activate_count = 1

        for obstacle in obstacles:
            obstacle.color = np.array([0, 1, 1, 1])
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
        return target.state.p_pos

    def estimate_target_pos_2(self, agent, target, coef=4.0):
        return target.prev_state.p_pos - (target.prev_state.speed * coef)

    def estimate_target_pos_3(self, agent, target, coef=2.0):
        return target.state.p_pos - (target.state.p_vel * coef)

    def estimate_target_pos_4(self, agent, target, coef=0.1):
        delta = target.state.p_pos - agent.state.p_pos
        mag = np.linalg.norm(delta)
        new_target = (
            target.state.p_pos - (coef * delta / mag) if mag > 0 else target.state.p_pos
        )
        return new_target

    def estimate_target_pos_5(self, agent, target, coef=0.2):
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

    def get_modulated_leader_callback(self):
        """Generates the modulated leader trajectory callback.

        Returns:
            function: Action callback function.
        """
        # phase accumulator for leader circle trajectory
        phase = float(randint(1, 100))
        # step counter for slow modulation frequency
        step_count = 0

        # goal activation tracking
        active_goal_idx = -1
        # countdown until next toggle
        toggle_countdown = randrange(30, 60)

        def cb(agent, world):
            nonlocal phase, step_count, active_goal_idx, toggle_countdown, self
            step_count += 1

            # --- 1. Leader Speed & Direction Modulation ---
            delta_phase = (
                1.0 + 1.2 * math.sin(step_count * 0.05) + np.random.normal(0, 0.2)
            )

            last_phase = phase
            phase += delta_phase

            act = Action()
            scale = 2 / 3

            # Calculate new position based on phase
            scaled_step = (phase / 50) * (2 * math.pi) * 0.75
            new_pos = (
                np.array([math.sin(scaled_step), math.cos(scaled_step)]) * scale
            )
            agent.state.p_pos = new_pos

            # Calculate speed based on delta phase
            last_scaled_step = (last_phase / 50) * (2 * math.pi) * 0.75
            last_new_pos = (
                np.array([math.sin(last_scaled_step), math.cos(last_scaled_step)])
                * scale
            )
            agent.state.speed = new_pos - last_new_pos

            act.u = np.zeros(2)

            # --- 2. Randomized & Mutually Exclusive Goal Activation ---
            goals = [ld for ld in world.landmarks if ld.type == "goal"]
            toggle_countdown -= 1
            if toggle_countdown <= 0:
                if active_goal_idx != -1:
                    goals[active_goal_idx].activate = False
                    goals[active_goal_idx].color = np.array([0, 1, 0, 0.75])
                    active_goal_idx = -1
                    toggle_countdown = randrange(15, 30)
                else:
                    active_goal_idx = randrange(0, len(goals))
                    for idx, goal in enumerate(goals):
                        if idx == active_goal_idx:
                            goal.activate = True
                            goal.color = np.array([1, 1, 0, 0.75])
                        else:
                            goal.activate = False
                            goal.color = np.array([0, 1, 0, 0.75])
                    toggle_countdown = randrange(40, 80)

            return act

        return cb

    def get_circle_leader_callback(self):
        """Generates the standard circular leader trajectory callback.

        Returns:
            function: Action callback function.
        """
        STEPS_FOR_FULL_ROTATION = 50 / 0.75
        step = randint(1, int(STEPS_FOR_FULL_ROTATION))

        def cb(agent, world):
            nonlocal step, self
            step += 1
            act = Action()

            # Circle
            scale = 2 / 3

            scaled_step = (step / 50) * (2 * math.pi) * 0.75
            new_pos = (
                np.array([math.sin(scaled_step), math.cos(scaled_step)]) * scale
            )
            agent.state.p_pos = new_pos

            last_scaled_step = ((step - 1) / 50) * (2 * math.pi) * 0.75
            last_new_pos = (
                np.array([math.sin(last_scaled_step), math.cos(last_scaled_step)])
                * scale
            )
            agent.state.speed = new_pos - last_new_pos

            act.u = np.zeros(2)

            return act

        return cb

