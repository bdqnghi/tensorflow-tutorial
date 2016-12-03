from random import seed
from random import random

def init_network(inputs,hidden,outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(inputs + 1)]} for i in range(hidden)]
	network.append(hidden_layer)
	output_layer =  [{'weights':[random() for i in range(inputs + 1)]} for i in range(outputs)]
	network.append(output_layer)
	return network

#seed(1)
network = init_network(2,1,2)
for layer in network:
	print(layer)