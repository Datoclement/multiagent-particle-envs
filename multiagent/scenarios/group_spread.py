import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.num_groups = 2
        self.num_landmarks = 3
        self.has_collision = False          # individual can/cannot overlap
        self.has_comm = False               # individual can/cannot communicate
        self.different_landmarks = False    # targets of different groups are/aren't different.
        # world.collaborative = True
        self.group_colours = [
            [0.35, 0.35, 0.85],
            [0.35, 0.85, 0.35]]

        assert (len(self.group_colours) == self.num_groups)

        # add agents
        num_agents = self.num_groups * self.num_landmarks
        world.agents = [Agent() for i in range(num_agents)] 
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d group %d' % (i, i // self.num_landmarks)
            agent.collide = self.has_collision
            agent.silent = not self.has_comm
            agent.size = 0.15

        # add landmarks
        total_landmarks =  \
                num_agents if self.different_landmarks else \
                self.num_landmarks
        world.landmarks = [Landmark() for i in range(total_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = ('landmark %d' % i) \
                    + (' group %d' % (i//self.land_marks) \
                    if self.different_landmarks else '')
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def get_group_id(self, agent):
        return int(agent.name.strip().split(' ')[-1])

    def reset_world(self, world):

        # properties
        #   agents
        for i, agent in enumerate(world.agents):
            gid = self.get_group_id(agent)
            agent.color = np.array(self.group_colours[gid])

        #   landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array(self.group_colours[i])/2. \
                    if self.different_landmarks else np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):

        # get group id of the agent
        group_id = self.get_group_id(agent)
        group = world.agents[
                ( group_id ) * self.num_landmarks : 
                ( group_id+1 ) * self.num_landmarks]
        targets = world.landmarks[
                ( group_id ) * self.num_landmarks : 
                ( group_id+1 ) * self.num_landmarks] \
                if self.different_landmarks \
                else world.landmarks

        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        for l in targets:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in group]
            rew -= min(dists)

        if self.has_collision:
            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent):
                        rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
                  [agent.state.p_vel] 
                + [agent.state.p_pos] 
                + entity_pos + entity_color + other_pos + comm)
