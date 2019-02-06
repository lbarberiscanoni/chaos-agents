from random import randint
from tensorforce.agents import PPOAgent, VPGAgent, DQNAgent
import numpy as np
from tqdm import tqdm
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--monkey", help="select a monkey type [ppo, vpg, dqn]")
parser.add_argument("--manager", help="select a manager type [ppo, vpg, dqn]")

args = parser.parse_args()


class Network:

	def __init__(self, servers, clients):
		self.clients = clients
		self.servers = servers

		self.graph = np.full((servers, clients), 0)

	def initializeGraph(self):
		for x in range(self.servers):
			for y in range(self.clients):
				self.graph[x][y] = randint(0, 1)
		
	def reward(self, vector, matrix):

		num_of_servers = len(vector)

		downClients = 0
		for val in vector:
			clientsConnectedToServer = np.count_nonzero(matrix[val] == 1)
			downClients += clientsConnectedToServer

		reward = 0
		if len(vector) > 0:
			reward = downClients / float(len(vector))
		else:
			reward = 0

		self.graph = matrix

		return reward

infrastructure = Network(20, 30)
infrastructure.initializeGraph()

if args.monkey == "ppo":
	monkey = PPOAgent(
	    states={"type":'float', "shape": infrastructure.graph.shape },
	    actions={
	    	str(i): dict(type="int", num_actions=infrastructure.servers) for i in range(int(infrastructure.servers * .1))
	    },
	    network=[
		    dict(type='flatten'),
		    dict(type="dense", size=32),
		   	dict(type="dense", size=32),
		   	dict(type="dense", size=32)
	    ],
	)
elif args.monkey == "dqn":
	monkey = DQNAgent(
	    states={"type":'float', "shape": infrastructure.graph.shape },
	    actions={
	    	str(i): dict(type="int", num_actions=infrastructure.servers) for i in range(int(infrastructure.servers * .1))
	    },
	    network=[
		    dict(type='flatten'),
		    dict(type="dense", size=32),
		   	dict(type="dense", size=32),
		   	dict(type="dense", size=32)
	    ],
	)
elif args.monkey == "vpg":
	monkey = VPGAgent(
	    states={"type":'float', "shape": infrastructure.graph.shape },
	    actions={
	    	str(i): dict(type="int", num_actions=infrastructure.servers) for i in range(int(infrastructure.servers * .1))
	    },
	    network=[
		    dict(type='flatten'),
		    dict(type="dense", size=32),
		   	dict(type="dense", size=32),
		   	dict(type="dense", size=32)
	    ],
	)

if args.manager == "ppo":
	manager = PPOAgent(
	    states={"type":'float', "shape": infrastructure.graph.shape },
	    actions={
	    	str(i): dict(type="int", num_actions=infrastructure.servers) for i in range(infrastructure.clients)
	    },
	    network=[
		    dict(type='flatten'),
		    dict(type="dense", size=32),
		   	dict(type="dense", size=32),
		   	dict(type="dense", size=32)
	    ],
	)
elif args.manager == "dqn":
	manager = DQNAgent(
	    states={"type":'float', "shape": infrastructure.graph.shape },
	    actions={
	    	str(i): dict(type="int", num_actions=infrastructure.servers) for i in range(infrastructure.clients)
	    },
	    network=[
		    dict(type='flatten'),
		    dict(type="dense", size=32),
		   	dict(type="dense", size=32),
		   	dict(type="dense", size=32)
	    ],
	)
elif args.manager == "vpg":
	manager = VPGAgent(
	    states={"type":'float', "shape": infrastructure.graph.shape },
	    actions={
	    	str(i): dict(type="int", num_actions=infrastructure.servers) for i in range(infrastructure.clients)
	    },
	    network=[
		    dict(type='flatten'),
		    dict(type="dense", size=32),
		   	dict(type="dense", size=32),
		   	dict(type="dense", size=32)
	    ],
	)

for i in tqdm(range(100000)):
	state = infrastructure.graph

	action_monkey = monkey.act(state).values()
	action_manager = manager.act(state)
	action_manager_matrix = np.full((infrastructure.servers, infrastructure.clients), 0)
	for item in action_manager.items():
		clientID = int(item[0])
		serverID = item[1]
		action_manager_matrix[serverID][clientID] = 1

	reward = infrastructure.reward(action_monkey, action_manager_matrix)

	monkey.observe(reward=reward, terminal=False)
	reward = reward * -1
	manager.observe(reward=reward, terminal=False)

rewards = []
for i in tqdm(range(100)):
	state = infrastructure.graph

	action_monkey = monkey.act(state).values()
	action_manager = manager.act(state)
	action_manager_matrix = np.full((infrastructure.servers, infrastructure.clients), 0)
	for item in action_manager.items():
		clientID = int(item[0])
		serverID = item[1]
		action_manager_matrix[serverID][clientID] = 1

	reward = infrastructure.reward(action_monkey, action_manager_matrix)

	rewards.append(reward)

	monkey.observe(reward=reward, terminal=False)
	reward = reward * -1
	manager.observe(reward=reward, terminal=False)

with open("results/client-server/" + args.monkey + "_" + args.manager + ".txt", "w") as f:
	pickle.dump(rewards, f)


