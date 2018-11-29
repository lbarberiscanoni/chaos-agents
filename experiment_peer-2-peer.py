import networkx as nx
from random import randint
from tensorforce.agents import PPOAgent, DQNAgent, VPGAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

class Network():

	def __init__(self, x):
		self.peers = [i for i in range(1, x + 1)]
		self.attempts = 0

	def initializeGraph(self):

		G = nx.Graph()

		for peer in self.peers:
			G.add_node(peer)

		for peer in self.peers:

			#this makes sure that each client is only connected to 1 server
			randNum = randint(0, len(self.peers) - 1)
			G.add_edge(self.peers[randNum], peer)

		self.graph = G
		self.attempts = 0

	def get_state(self):

		# matrix = nx.laplacian_matrix(G)
		# print(matrix)
		matrix = nx.to_numpy_matrix(self.graph)

		return matrix

	def shutdown(self, vector):

		self.attempts = self.attempts + 1

		selectedServers = [x for x in vector if x != 0]

		downClients = 0
		for serverID in selectedServers:
			clientsConnectedToTheServer = [x for x in self.graph.edges if serverID in x]
			downClients += len(clientsConnectedToTheServer)

		reward = 0
		if len(selectedServers) > 0:
			reward = downClients / float(len(selectedServers))
		else:
			reward = 0

		return reward

	def monkey(self, vector): 
		selectedServers = [x for x in vector if x != 0]

		downClients = 0
		for serverID in selectedServers:
			clientsConnectedToTheServer = [x for x in self.graph.edges if serverID in x]
			downClients += len(clientsConnectedToTheServer)

		reward = 0
		if len(selectedServers) > 0:
			reward = downClients / float(len(selectedServers))
		else:
			reward = 0

		return reward


infrastructure = Network(30)
infrastructure.initializeGraph()

print("graph initalized")

# Create a Proximal Policy Optimization agent_ppo
agent_ppo = PPOAgent(
    states={"type":'float', "shape": infrastructure.get_state().shape },
    actions={
    	str(i): dict(type="int", num_actions=len(infrastructure.peers)) for i in range(int(len(infrastructure.peers) * .1))
    },
    network=[
	    dict(type='flatten'),
	    dict(type="dense", size=32),
    ],
)

# Create a Deep Q Network 
agent_dqn = DQNAgent(
    states={"type":'float', "shape": infrastructure.get_state().shape },
    actions={
    	str(i): dict(type="int", num_actions=len(infrastructure.peers)) for i in range(int(len(infrastructure.peers) * .1))
    },
    network=[
	    dict(type='flatten'),
	    dict(type='dense', size=32,activation='relu'),
    ],
)

# Create a Vanilla Policy Gradient
agent_vpg = VPGAgent(
    states={"type":'float', "shape": infrastructure.get_state().shape },
    actions={
    	str(i): dict(type="int", num_actions=len(infrastructure.peers)) for i in range(int(len(infrastructure.peers) * .1))
    },
    network=[
	    dict(type='flatten'),
	    dict(type='dense', size=32,activation='relu'),
    ],
)

#agent_ppo.restore_model("results/client-server")

print("agents made")

monkey = []
rl_ppo = []
rl_dqn = []
rl_vpg = []

#training
for i in tqdm(range(5000)):
	infrastructure.initializeGraph()
	while infrastructure.attempts < len(infrastructure.peers):
		#agent_ppo actions
		state = infrastructure.get_state()

		action = agent_ppo.act(state)
		action = action.values()

		#print("ai", action)
		reward = infrastructure.shutdown(action)

		if infrastructure.attempts < infrastructure.peers:
			agent_ppo.observe(reward=reward, terminal=False)
		else:
			agent_ppo.observe(reward=reward, terminal=True)

		rl_ppo.append(reward)

		#dqn agent
		action = agent_dqn.act(state)
		action = action.values()

		reward = infrastructure.monkey(action)

		if infrastructure.attempts < infrastructure.peers:
			agent_dqn.observe(reward=reward, terminal=False)
		else:
			agent_dqn.observe(reward=reward, terminal=True)

		rl_dqn.append(reward)

		#trpo agent
		action = agent_vpg.act(state)
		action = action.values()

		reward = infrastructure.monkey(action)

		if infrastructure.attempts < infrastructure.peers:
			agent_vpg.observe(reward=reward, terminal=False)
		else:
			agent_vpg.observe(reward=reward, terminal=True)

		rl_vpg.append(reward)


		#monkey
		action = [randint(1, len(infrastructure.peers) - 1) for x in range(int(len(infrastructure.peers) * .2))]
		reward = infrastructure.monkey(action)


		#print("monkey", action)

		monkey.append(reward)


x = [x for x in range(len(rl_ppo[-100:]))]

fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(x, monkey[-100:], label='chaos monkey')


# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax.plot(x, rl_ppo[-100:], label='PPO')

# Using plot(..., dashes=...) to set the dashing when creating a line
line3, = ax.plot(x, rl_dqn[-100:], label='DQN')

# Using plot(..., dashes=...) to set the dashing when creating a line
line4, = ax.plot(x, rl_vpg[-100:], label='VPG')

print("ppo", np.median(rl_ppo[-100:]), mean_confidence_interval(rl_ppo[-100:]))
print("dqn", np.median(rl_dqn[-100:]), mean_confidence_interval(rl_dqn[-100:]))
print("vpg", np.median(rl_vpg[-100:]), mean_confidence_interval(rl_vpg[-100:]))
print("monkey", np.median(monkey[-100:]), mean_confidence_interval(monkey[-100:]))


ax.legend()
plt.savefig("peer-2-peer.png")


print(infrastructure.get_state().shape)
