import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pickle

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

options = ["ppo", "vpg", "dqn"]

for monkey in options:
# monkey = "ppo"
	data = {}
	for option in options:
		with open("results/peer-2-peer/" + monkey + "_" + option + ".txt", "r") as f:
			txt = pickle.load(f)
			data[option] = txt

	print(data)

	fig, ax = plt.subplots()

	x = [x for x in range(len(txt))]

	for option in options:
		ax.plot(x, data[option], label=option)

	ax.legend()
	plt.savefig("results/peer-2-peer/gan-" + monkey + ".png")

	for option in options:
		print(option, np.median(data[option]), mean_confidence_interval(data[option]))