from random import randint, randrange

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
        self.nb_agents = 4 # including the leader
        self.nb_goals = 3
        self.nb_obstacles = 0

    def make_world(self):
        world = World()

        # add agents
        world.agents = [Agent() for _ in range(self.nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = i
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
                agent.color = [.2 * i, 0, 1, 0.75]

        goals = [ld for ld in world.landmarks if ld.type == 'goal']
        obstacles = [ld for ld in world.landmarks if ld.type == 'obstacle']

        for i, goal in enumerate(goals):
            goal.color = np.array([1, 1, 0, 0.75])
            goal.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            goal.state.p_vel = np.zeros(world.dim_p)
            goal.color = np.array([0, 1, 0, 0.75])
            goal.activate = False
            goal.activate_time = (i+1) * 30 + randrange(5, 10)
            goal.activate_count = 1
            
        for obstacle in obstacles:
            obstacle.color = np.array([0, 1, 1, 1])
            obstacle.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)

    def find_entity_by_name(self, world, name:str):
        return next((e for e in world.entities if e.name == name), None)
    
    def find_agent_by_id(self, world, id:int):
        return next((a for a in world.agents if a.id == id), None)

    # def estimate_target_pos(self, agent, target, coef=2.):
    #     return target.prev_state.p_pos - (target.prev_state.speed * coef)
    
    # def estimate_target_pos(self, agent, target, coef=2.):
    #     return target.state.p_pos - (target.state.p_vel * coef)
    
    def estimate_target_pos(self, agent, target, coef=2.):
        return target.state.p_pos

    # def estimate_target_pos(self, agent, target, coef=.1):
    #     delta = target.state.p_pos - agent.state.p_pos
    #     mag = np.linalg.norm(delta)
    #     new_target = target.state.p_pos - (coef * delta / mag)
    #     new_mag = dist(target.state.p_pos, new_target)

    #     # print(mag, new_mag)
    #     # new_angle = self.get_angle(delta, new_target - agent.state.p_pos)
    #     # assert(math.isclose(new_mag, coef))
    #     # assert(math.isclose(new_angle, 0))
    #     return new_target
    
    # def estimate_target_pos(self, agent, target, coef=.2):
    #     p_vel_mag = np.linalg.norm(target.state.p_vel)
    #     p_vel_u = target.state.p_vel / p_vel_mag
    #     new_vel = -coef * p_vel_u
    #     new_pos = target.state.p_pos + new_vel

    #     # new_mag = dist(target.state.p_pos, new_pos)
    #     # new_angle = self.get_angle(target.state.p_vel, new_vel)
    #     # # print(new_angle, new_mag)
    #     # assert(math.isclose(new_mag, coef, rel_tol= 0.01))
    #     # assert(math.isclose(new_angle, math.pi, rel_tol = 0.01))
    #     return new_pos

    def get_angle(self, v1, v2):
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        v1_u = v1 / v1_mag if v1_mag > 0 else np.array([0, 0])
        v2_u = v2 / v2_mag if v2_mag > 0 else np.array([0, 0])

        dv_dot = np.dot(v1_u, v2_u)
        dv_dot = 0 if np.isnan(dv_dot) else dv_dot
        dv_cross = np.cross(v1_u, v2_u)
        dv_cross = 0 if np.isnan(dv_cross) else dv_cross

        dv_mag = abs(v1_mag - v2_mag)
        dv_angle = np.arctan2(dv_cross, dv_dot)
        return dv_angle, dv_mag

    def cos_sim(self, v1, v2):
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        cos_sim = np.dot(v1, v2)/(v1_norm*v2_norm)
        return cos_sim
    
    def sigmoid(self, x):
        var = 4
        shift = 9
        return 1 / (1 + np.exp(- var + (x*shift)))

    def fix_agent_vel(self, agent):
        if agent.state.p_vel[0] == 0 and agent.state.p_vel[1] == 0 and hasattr(agent.state, "speed"):
            agent.state.p_vel = agent.state.speed

    def reward(self, agent, world):
        target_id = agent.id-1
        target = self.find_agent_by_id(world, target_id)

        # if the goal is activated, try to get it
        landmark = self.find_entity_by_name(world, f"Goal {target_id}")
        if landmark and landmark.activate:
            d = dist(agent.state.p_pos, landmark.state.p_pos)
            reward = -math.log(d)

        # else follow the leader
        else:
            target_pos = self.estimate_target_pos(agent, target)
            d = dist(agent.state.p_pos, target_pos)
            angle, mag = self.get_angle(target.state.p_vel, agent.state.p_vel)
            reward = .7 * -math.log(d) + .15 * -math.log(abs(angle)) + .15 * -math.log(mag)
        return reward

    def observation(self, agent, world):
        dx, dy, dv_mag, dv_angle = 0, 0, 0, 0
        lm_dx, lm_dy, lm_act = 0, 0, 0
        target_id = agent.id-1

        # target distance
        target = self.find_agent_by_id(world, target_id)
        if target:
            self.fix_agent_vel(target)
            # delta x/y
            dx = target.state.p_pos[0] - agent.state.p_pos[0]
            dy = target.state.p_pos[1] - agent.state.p_pos[1]
            
            # delta vel
            dv_angle, dv_mag = self.get_angle(target.state.p_vel, agent.state.p_vel)
            
        # agent's goal
        lm = self.find_entity_by_name(world, f"Goal {target_id}")
        if lm:
            lm_dx = lm.state.p_pos[0] - agent.state.p_pos[0]
            lm_dy = lm.state.p_pos[1] - agent.state.p_pos[1]
            lm_act = int(lm.activate == True)
        
        # return the complete state
        return np.array([dx, dy, dv_angle, dv_mag, lm_dx, lm_dy, lm_act])

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