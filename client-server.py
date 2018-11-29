import networkx as nx
from random import randint
from tensorforce.agents import PPOAgent
from tqdm import tqdm


class Network():

	def __init__(self, x, y):
		self.servers = [i for i in range(1, x + 1)]
		self.clients = [i for i in range(100, y + 101)]
		self.attempts = 0

	def initializeGraph(self):

		G = nx.Graph()

		for server in self.servers:
			G.add_node(server)

		for client in self.clients:

			G.add_node(client)

			#this makes sure that each client is only connected to 1 server
			randNum = randint(0, len(self.servers) - 1)
			G.add_edge(self.servers[randNum], client)

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


infrastructure = Network(20, 40)
infrastructure.initializeGraph()

print("graph initalized")

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states={"type":'float', "shape": infrastructure.get_state().shape },
    actions={
    	str(i): dict(type="int", num_actions=len(infrastructure.servers)) for i in range(int(len(infrastructure.servers) * .1))
    },
    network=[
	    dict(type='flatten'),
	    dict(type="dense", size=32),
    ],
)

print("agent made")
#training
for i in tqdm(range(1000000)):
	infrastructure.initializeGraph()
	while infrastructure.attempts < len(infrastructure.servers):
		#print("epoch", str(i), "attempt", str(infrastructure.attempts))
		state = infrastructure.get_state()

		action = agent.act(state)
		action = action.values()

		reward = infrastructure.shutdown(action)

		if infrastructure.attempts < infrastructure.servers:
			agent.observe(reward=reward, terminal=False)
		else:
			agent.observe(reward=reward, terminal=True)

		#print(action, reward)

agent.save_model("results/client-server")

print(infrastructure.get_state().shape)
