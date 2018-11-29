from tensorforce.agents import RandomAgent as chaos_monkey

config = chaos_monkey(
	states=dict(type="int", shape=(2, 2)), 
	actions=dict(type="int", shape=(1), num_actions=(3))
)

print config