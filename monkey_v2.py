import subprocess
from random import randint

pList = subprocess.check_output("ps", shell=True)

serverList = [x for x in pList.split("\n") if "server.py" in x]

pID = serverList[randint(0, len(serverList) - 1)].split(" ")[1]

subprocess.call("kill " + str(pID), shell=True)