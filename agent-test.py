import networkx as nx
from random import randint
from tensorforce.agents import PPOAgent

agent = PPOAgent(
    states={"type":'float', "shape": (51,51) },
    actions={
    	str(i): dict(type="int", num_actions=20) for i in range(int(20 * .1))
    },
    network=[
	    dict(type='flatten'),
	    dict(type="dense", size=32),
	   	dict(type="dense", size=32),
	   	dict(type="dense", size=32)
    ],
)

agent.restore_model("results/")