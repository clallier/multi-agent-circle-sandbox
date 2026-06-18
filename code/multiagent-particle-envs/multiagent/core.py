from random import randrange
from multiagent import palette
import numpy as np


class EntityState(object):
    """Physical and external base state of all entities.

    This class encapsulates physical state attributes shared by all simulator entities.

    Attributes:
        p_pos (np.ndarray): Physical 2D position in space.
        p_vel (np.ndarray): Physical 2D velocity vector.
    """

    def __init__(self):
        self.p_pos = None
        self.p_vel = None


class AgentState(EntityState):
    """State of agents, extending EntityState with communication/mental properties.

    This class encapsulates both the physical attributes and the communicative/mental state of an agent.

    Attributes:
        c (np.ndarray): Communication utterance signal.
    """

    def __init__(self):
        super(AgentState, self).__init__()
        self.c = None


class Action(object):
    """Encapsulates the action choices of an agent.

    This class represents action commands executed by agents in physical and communication domains.

    Attributes:
        u (np.ndarray): Physical action command.
        c (np.ndarray): Communication action command.
    """

    def __init__(self):
        self.u = None
        self.c = None


class Entity(object):
    """Properties and state of a physical world entity.

    Base class for all physical entities present in the simulation environment.

    Attributes:
        name (str): Unique identifying name.
        size (float): Radius size of the entity.
        movable (bool): Indicates if the entity is movable.
        collide (bool): Indicates if the entity collides with others.
        density (float): Material density affecting mass computation.
        color (np.ndarray): Color of the entity.
        type (str): Type categorization, e.g. "landmark", "goal", "obstacle".
        max_speed (float): Maximum allowed velocity speed limit.
        accel (float): Acceleration limit.
        state (EntityState): Core state representation.
        initial_mass (float): Base mass of the entity.
    """

    def __init__(self):
        self.name = ""
        self.size = 0.050
        self.movable = False
        self.collide = True
        self.density = 25.0
        self.color = None
        self.type = "landmark"
        self.max_speed = None
        self.accel = None
        self.state = EntityState()
        self.initial_mass = 1.0

    @property
    def mass(self):
        """Computes the mass of the entity.

        Returns:
            float: Initial mass of the entity.
        """
        return self.initial_mass


class Landmark(Entity):
    """Properties of landmark entities.

    A stationary landmark in the world that can represent a target, goal, or obstacle.

    Attributes:
        activate (bool): Current activation status of the landmark.
        deactivation_time (int): Steps before deactivating a landmark.
        next_activate_time_min (int): Minimum steps before next activation.
        next_activate_time_max (int): Maximum steps before next activation.
        activate_time (int): Target step counter at which the landmark state toggles.
        activate_count (int): Counter tracking steps since last state change.
        activate_prob (float): Probability parameter for activation state.
    """

    def __init__(self):
        super(Landmark, self).__init__()
        self.activate = False
        self.deactivation_time = 30
        self.next_activate_time_min = 5
        self.next_activate_time_max = 10
        self.activate_time = randrange(
            self.next_activate_time_min, self.next_activate_time_max
        )
        self.activate_count = 1
        self.activate_prob = 0.1
        self.type = "landmark"
        self.color = np.array(palette.TARGET_BASE)

    def goal_activate(self):
        """Ticks the landmark state and toggles goal activation color based on step intervals.

        How it works:
            If this landmark is a goal, ticks self.activate_count. Upon reaching
            self.activate_time, toggles the activation state, changes the target color,
            and resets the activation timing window.
        """
        if self.type != "goal":
            return
        if self.activate_count > 0 and self.activate_count < self.activate_time:
            self.activate_count += 1
        elif self.activate_count >= self.activate_time:
            self.activate_count = 1
            if not self.activate:
                self.activate = True
                self.activate_time = randrange(
                    self.next_activate_time_min, self.next_activate_time_max
                )
                self.color = np.array(palette.TARGET_ACTIVATED)
            else:
                self.activate = False
                self.activate_time = self.deactivation_time
                self.color = np.array(palette.TARGET_BASE)


class Agent(Entity):
    """Properties of agent entities.

    An active physical agent in the multi-agent sandbox.

    Attributes:
        id (int): Numerical identifier.
        movable (bool): Indicates if the agent can move (True by default).
        silent (bool): Communication signal restriction.
        blind (bool): Observation restriction.
        u_noise (float): Physical motor noise.
        c_noise (float): Communication noise.
        u_range (float): Control output bounds range.
        state (AgentState): Current agent state.
        action (Action): Last selected action.
        action_callback (callable): Scripted behavior function.
        prev_state (AgentState): Historical previous agent state.
    """

    def __init__(self):
        super(Agent, self).__init__()
        self.id = -1
        self.movable = True
        self.silent = False
        self.blind = False
        self.u_noise = None
        self.c_noise = None
        self.u_range = 1.0
        self.state = AgentState()
        self.action = Action()
        self.action_callback = None
        self.prev_state = AgentState()


class World(object):
    """Multi-agent simulation environment world.

    Manages all agents, landmarks, laws of physics, collisions, and state steps.

    Attributes:
        agents (list[Agent]): Active agents list.
        landmarks (list[Landmark]): Stationary landmarks list.
        dim_c (int): Dimensions of communication channel.
        dim_p (int): Dimensions of position channel (2D).
        dim_color (int): Dimensions of color components (3).
        dt (float): Simulation timestep duration.
        damping (float): Friction damping coefficient.
        contact_force (float): Collision repulsion magnitude force.
        contact_margin (float): Penetration margin distance threshold.
    """

    def __init__(self):
        self.agents = []
        self.landmarks = []
        self.dim_c = 0
        self.dim_p = 2
        self.dim_color = 3
        self.dt = 0.1
        self.damping = 0.25
        self.contact_force = 5e2
        self.contact_margin = 1e-3

    @property
    def entities(self):
        """Returns all entities in the simulation world.

        Returns:
            list[Entity]: Concatenated list of agents and landmarks.
        """
        return self.agents + self.landmarks

    @property
    def policy_agents(self):
        """Returns agents controllable by external neural network/RL policies.

        Returns:
            list[Agent]: All agents without a scripted callback.
        """
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        """Returns agents controlled by hardcoded world script callbacks.

        Returns:
            list[Agent]: All agents with a scripted callback.
        """
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        """Advances the simulation world by one physical step.

        How it works:
            1. Computes actions for scripted agents.
            2. Applies physical actions and environment collision forces.
            3. Integrates state equations to calculate new positions and velocities.
            4. Updates internal agent properties and landmark activation.
        """
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        p_force = [None] * len(self.entities)
        p_force = self.apply_action_force(p_force)
        p_force = self.apply_environment_force(p_force)
        self.integrate_state(p_force)
        for agent in self.agents:
            self.update_agent_state(agent)
        for landmark in self.landmarks:
            landmark.goal_activate()

    def apply_action_force(self, p_force):
        """Gathers and maps agent action controls to force vectors.

        Args:
            p_force (list): Initialized force list to modify.

        Returns:
            list: Force list populated with action vectors and motor noise.
        """
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
        return p_force

    def apply_environment_force(self, p_force):
        """Gathers physical environmental collision forces acting on all entities.

        Args:
            p_force (list): Populated action force list.

        Returns:
            list: Combined action and collision force list.
        """
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    def integrate_state(self, p_force):
        """Integrates physical acceleration, velocity, and position equations.

        Args:
            p_force (list): Total force vectors acting on entities.
        """
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        """Updates communication status and noise filters of an agent.

        Args:
            agent (Agent): Agent entity to update.
        """
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    def get_collision_force(self, entity_a, entity_b):
        """Computes contact forces between two colliding/intersecting entities.

        Args:
            entity_a (Entity): First entity.
            entity_b (Entity): Second entity.

        Returns:
            list[np.ndarray | None]: Force vectors exerted on [entity_a, entity_b].
        """
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]
        if entity_a is entity_b:
            return [None, None]
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity_a.size + entity_b.size
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
