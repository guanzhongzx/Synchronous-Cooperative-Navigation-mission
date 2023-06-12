import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import pickle

SaveIt = True  ### True #False

RE=2
gameN=-2
nce = [[]]###抓到的 goodagents
NC=0
NC_episode=0
nceB = [[]]###抓到的 bule goodagents
NCB=0
NCB_episode=0
nre = [[]]###到达的 foods
NR=0
NR_episode=0

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 4
        #world.damping = 1
        num_good_agent1 = 6 ####blue
        num_good_agent2 = 0 ###green
        num_good_agents = num_good_agent1 + num_good_agent2 ###
        num_adversaries = 0 ###red
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2
        num_food = 1

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = False#True if i == 0 else False
            agent.silent = False#True if i > 0 else False
            agent.adversary = True if i < num_adversaries else False
            agent.green = True if i >= num_adversaries+num_good_agent1 else False ###
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.12
            landmark.boundary = False
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False

        world.landmarks += world.food
        #world.landmarks += self.set_boundaries(world)  # world boundaries now penalized with negative reward
        # make initial conditions
        self.reset_world(world)
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.55, 0.95]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            agent.color += np.array([0, 0.4, -0.4]) if agent.green else np.array([0, 0, 0])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])

        # set random initial states

        global RE, gameN, NC ,NC_episode, nce, NCB ,NCB_episode, nceB, NR, NR_episode, nre ###我加的
        RE -= 1 ###我加的
        gameN += 1
        print('game:',gameN)
        if RE < 0: ###回合结束，新回合开始标志
            RE = 0
            # print('Number of Catched Agents in this Episode =', NC_episode)
            print('Number of Catched Agents =', NC)  ###
            # print('Number of Catched BuleAgents in this Episode =', NCB_episode)
            print('Number of Catched BuleAgents =', NCB)  ###
            # print('Number of Reached Food in this Episode =', NR_episode)
            print('Number of Reached Foods =', NR)  ###

            if SaveIt:
                if len(nce[-1]) < 1000:  ###"--save-rate", type=int, default=1000,
                    nce[-1].append(NC_episode)
                else:
                    nce.append([NC_episode])
                #########
                file_name = './learning_curves_tt/' + 'Number' + '_AgentsCatched.pkl'  ###
                with open(file_name, 'wb') as fp:
                    pickle.dump(nce, fp)  ###存储每回合抓捕次数
                # #########

                if len(nceB[-1]) < 1000:  ###"--save-rate", type=int, default=1000,
                    nceB[-1].append(NCB_episode)
                else:
                    nceB.append([NCB_episode])
                #########
                file_name = './learning_curves_tt/' + 'Number' + '_BuleAgentsCatched.pkl'  ###
                with open(file_name, 'wb') as fp:
                    pickle.dump(nceB, fp)  ###存储每回合抓捕蓝agent次数
                #########

                if len(nre[-1]) < 1000:  ###"--save-rate", type=int, default=1000,
                    nre[-1].append(NR_episode)
                else:
                    nre.append([NR_episode])
                #########
                file_name = './learning_curves_tt/' + 'Number' + '_FoodsReached.pkl'  ###
                with open(file_name, 'wb') as fp:
                    pickle.dump(nre, fp)  ###存储每回合到目标次数
                    #########

            NC_episode = 0 ###存储每回合抓捕次数
            NCB_episode= 0 ###存储每回合抓捕蓝agent次数
            NR_episode = 0  ###每回合到目标次数

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_pos = 1.5 * agent.state.p_pos if not agent.green else 0.7 * agent.state.p_pos
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    # return all agents that are not adversaries
    def good_agent1(self, world):  ####
        return [agent for agent in world.agents if not (agent.adversary or agent.green)]  ####
    def good_agent2(self, world):  ####
        return [agent for agent in world.agents if agent.green]  ####

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        if agent.adversary:
            main_reward = self.adversary_reward(agent, world)
        elif agent.green:
            main_reward = self.agent2_reward(agent, world)
        else:
            main_reward = self.agent1_reward(agent, world)
        return main_reward

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False

    def agent1_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        good_agent1 = self.good_agent1(world)
        for landmark in world.landmarks:
            if self.is_collision(landmark, agent):
                rew -= 2
        # adversary = self.adversaries(world)
        # if agent.collide:
        #     for ad in adversary:
        #         if self.is_collision(ad, agent):
        #             rew -= 5
        # ###
        # rew += 2 * min([np.sqrt(np.sum(np.square(agent.state.p_pos - ad.state.p_pos))) for ad in adversary])

        c = []
        for a in good_agent1:
            dists = min([np.sqrt(np.sum(np.square(a.state.p_pos - food.state.p_pos))) for food in world.food])
            c.append(dists)
        # print(c)
        c_std = np.std(c)###计算距离的均方差
        # print(c_std)
        rew -= 5 * c_std

        global NR, NR_episode
        for food in world.food:
            if self.is_collision(agent, food):
                rew += 2.5  ###2
                NR += 1
                NR_episode += 1
            rew -= 2 * min([np.sqrt(np.sum(np.square(a.state.p_pos - food.state.p_pos))) for a in good_agent1])

        return rew

    def agent2_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        adversaries = self.adversaries(world)
        good_agent1 = self.good_agent1(world)
        for adv in adversaries:
            jvli = np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))  # 远离对手加奖励 ###
            rew += 2.545 * np.exp(-np.square((jvli - 0.56)) * 20) + 0.9 * (0.78 - np.square(jvli - 1.063))  # 加奖励 # rew += 0.1463 * np.exp(-np.square((jvli-1.02))*13.1)+0.1*(1.464-np.square(jvli-1.335))#

        if agent.collide:
            for adv in adversaries:
                if self.is_collision(adv, agent):
                    rew -= 5  # 被抓减奖励

        dis = min([np.sqrt(np.sum(np.square(agent.state.p_pos - a1.state.p_pos))) for a1 in good_agent1])
        rew += 0.5 * dis # 远离蓝agent1
        return rew


    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        agents = self.good_agent1(world) + self.good_agent2(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents])
        global NC, NC_episode, NCB, NCB_episode###我加的
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 5
                        NC += 1  ###
                        NC_episode += 1  ###
                        # print('NC=', NC)    ####
                        if not ag.green:
                            NCB += 1 ###
                            NCB_episode += 1 ###
                            # print('NCB=', NCB)    ####
        return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)



