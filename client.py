# Python TCP Client A
import socket 
import sys
import subprocess
from random import randint

pList = subprocess.check_output("ps", shell=True)

serverList = [x.split(" ")[len(x.split(" ")) - 1] for x in pList.split("\n") if "server.py" in x]
print(serverList)


host = socket.gethostname() 
port = int(serverList[randint(0, len(serverList) -1)])
print(port)
BUFFER_SIZE = 2000 
MESSAGE = str(sys.argv[1])
 
tcpClientA = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
tcpClientA.connect((host, port))

while MESSAGE != 'exit':
    tcpClientA.send(MESSAGE)     
    data = tcpClientA.recv(BUFFER_SIZE)
    print " Client received data:", data

tcpClientA.close() 