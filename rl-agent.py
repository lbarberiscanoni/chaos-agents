from tensorforce.agents import PPOAgent

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states={"type":'float', "shape":(2,)},
    actions= {
    	str(i): dict(type="int", num_actions=2) for i in range(5)
    },
    network=[
        dict(type='dense', size=32),
        dict(type='dense', size=32),
    ],
)

state = [2, 1]

for i in range(50):
	action = agent.act(state)

	print(action.values())