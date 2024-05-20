from random import randint

import numpy as np
import math
from multiagent.core import World, Agent, Landmark, Action
from multiagent.scenario import BaseScenario


def dist(a, b):
    def quick_norm(vec):
        """
        Norme rapide d'un vecteur 2D.
        """
        return math.hypot(vec[0], vec[1])

    return quick_norm(a - b)


class Scenario(BaseScenario):

    def __init__(self):
        self.nb_agents = 2 # including the leader
        self.nb_goals = 0
        self.nb_obstacles = 0

    def make_world(self):
        world = World()

        # add agents
        world.agents = [Agent() for _ in range(self.nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.color = [1, 1, 0, 1]
            agent.silent = True
            agent.size = 0.1

        # add goals
        goals = [Landmark() for _ in range(self.nb_goals)]
        for i, goal in enumerate(goals):
            goal.name = 'Goal %d' % i
            goal.type = 'goal'
            goal.color = [1, 0, 0, 0.75]
            goal.collide = False
            goal.movable = False
            goal.size = 0.2


        # add obstacles
        obstacles = [Landmark() for _ in range(self.nb_obstacles)]
        for i, obstacle in enumerate(obstacles):
            obstacle.name = 'Obstacle %d' % i
            obstacle.type = 'obstacle'
            obstacle.collide = True
            obstacle.movable = False
            obstacle.color = [1, 0, 1, 1]
            obstacle.size = 0.1

        world.landmarks = goals + obstacles

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
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
                agent.color = [0, 0, 1, 0.75]

        goals = [ld for ld in world.landmarks if ld.type == 'goal']
        obstacles = [ld for ld in world.landmarks if ld.type == 'obstacle']

        for goal in goals:
            goal.color = np.array([1, 1, 0, 0.75])
            goal.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            goal.state.p_vel = np.zeros(world.dim_p)

        for obstacle in obstacles:
            obstacle.color = np.array([0, 1, 1, 1])
            obstacle.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # print(world)
        # print(agent)
        a_dist = dist(agent.state.p_pos, world.agents[0].state.p_pos)
        # reward = max(1 - a_dist *.5, 0.0001)
        reward = -math.log(a_dist)
        return reward

    def observation(self, agent, world):
        # TODO: validate input indexes
        dx = world.agents[0].state.p_pos[0] - agent.state.p_pos[0]
        dy = world.agents[0].state.p_pos[1] - agent.state.p_pos[1]
        return np.array([dx, dy])

    def create_leader_callback(self):
        # this functions defines the trajectory of the leader to be followed;
        # it is called at each env step to compute its new position
        # tip : you can use this function to implement the magnet

        STEPS_FOR_FULL_ROTATION = 50 / 0.75
        step = randint(1, int(STEPS_FOR_FULL_ROTATION))

        def cb(agent, world):
            nonlocal step, self
            step += 1
            act = Action()

            # Circle
            scale = 2 / 3

            scaled_step = (step / 50) * (2 * math.pi) * 0.75
            new_pos = np.concatenate(
                [
                    np.array([math.sin(scaled_step), math.cos(scaled_step)]) * scale
                ]
            )
            agent.state.p_pos = new_pos

            last_scaled_step = ((step - 1) / 50) * (2 * math.pi) * 0.75
            last_new_pos = np.concatenate(
                [
                    np.array([math.sin(last_scaled_step), math.cos(last_scaled_step)]) * scale
                ]
            )

            agent.state.speed = (new_pos - last_new_pos)

            act.u = np.zeros(2)

            return act

        return cb
