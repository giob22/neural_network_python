import random
import numpy as np
from neural_network.nn_engine import neural_network
from neural_network.nn_layer import *

p = neural_network([(8,sigmoid), (100, relu)],2,1,0.2,sigmoid)

for _ in range(50000):
    n1 = np.random.randint(0,2)
    n2 = np.random.randint(0,2)
    # The correct XOR logic: (n1 or n2) and not (n1 and n2)
    p.feedback([n1,n2],[(n1 or n2) and not (n1 and n2)])

print(f"[1,1] → {p.feedforward([1,1])}")
print(f"[1,0] → {p.feedforward([1,0])}")
print(f"[0,1] → {p.feedforward([0,1])}")
print(f"[0,0] → {p.feedforward([0,0])}")
