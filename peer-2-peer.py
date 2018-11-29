import networkx as nx
from random import randint
from tensorforce.agents import PPOAgent

class Network():

	def __init__(self, x):
		self.peers = [i for i in range(1, x + 1)]
		self.attempts = 0

	def initializeGraph(self):

		G = nx.Graph()

		for peer in self.peers:
			G.add_node(server)

		for peer in self.peers:

			#this makes sure that each client is only connected to 1 server
			randNum = randint(0, len(self.peers) - 1)
			G.add_edge(self.peers[randNum], peers)

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


infrastructure = Network(20)
infrastructure.initializeGraph()

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states={"type":'float', "shape": infrastructure.get_state().shape },
    actions={
    	str(i): dict(type="int", num_actions=len(infrastructure.peers)) for i in range(int(len(infrastructure.peers) * .1))
    },
    network=[
	    dict(type='flatten'),
	    dict(type="dense", size=32),
	   	dict(type="dense", size=32),
	   	dict(type="dense", size=32)
    ],
)

#training
for i in range(100):
	infrastructure.initializeGraph()
	while infrastructure.attempts < len(infrastructure.peers):
		print("epoch", str(i), "attempt", str(infrastructure.attempts))
		state = infrastructure.get_state()

		action = agent.act(state)
		action = action.values()

		reward = infrastructure.shutdown(action)

		if infrastructure.attempts < infrastructure.peers:
			agent.observe(reward=reward, terminal=False)
		else:
			agent.observe(reward=reward, terminal=True)

		print(action, reward)