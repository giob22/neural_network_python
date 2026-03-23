import numpy as np
import math as m


def sigmoid(x):
    # Clip per evitare overflow in np.exp
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def d_sigmoid(x):
    return x * (1 - x)


class neural_network:

    def __init__(self, n_layer, n_input, n_output, lr, hidden_size=8):
        self.lr = lr
        self.weights = []
        self.bias = []
        size = hidden_size
        print("-"*20 + "CARATTERISTICHE DELLA RETE NEURALE" + "-"*20)
        print(f"{'#layer:':<25} {n_layer:<10}\n{'dimensione input:':<25} {n_input:<10}\n{'dimensione output:':<25} {n_output:<10}\n{'dimensione hidden layer:':<25} {size:<10}")
        print("-"*20 + "-" * 34 + "-"*20)
        self.bias.append(np.random.uniform(-0.5,0.5, size=(size,1)))
        for i in range(n_layer - 2):
            t = np.random.uniform(-0.5,0.5,size=(size,size))
            self.weights.append(t)
            t = np.random.uniform(-0.5,0.5, size=(size,1)) # hidden_n , forse output_n
            self.bias.append(t)
        
        self.weights.insert(0, np.random.uniform(-0.5,0.5,size=(size, n_input)))
        


        self.bias.append(np.random.uniform(-0.5,0.5, size=(n_output, 1)))
        if n_layer > 1:
            self.weights.append(np.random.uniform(-0.5,0.5, size=(n_output, size)))

        self.n_input= n_input
        self.n_output = n_output
    
    def prediction(self, input):
        if isinstance(input, np.ndarray):

            # verifico la shape che deve essere del tipo (n,1)
            if input.ndim != 2 or input.shape[1] != 1:
                res = input.reshape(-1,1)
            else:
                res = input
        else:
            # nel caso fosse un array
            res = np.array(input).reshape(-1,1)

        hidden_outputs = []
        for i in range(len(self.weights)):
            res = self.weights[i] @ res + self.bias[i]
            res = sigmoid(res)
            hidden_outputs.append(res)
        hidden_outputs = hidden_outputs[:-1] # tolgo l'ultimo hidden_output che corrisponde alla predizione
        return (res, hidden_outputs)

    def backpropagation_hidden(self, hidden_outputs, output_error):
        hidden_error = output_error
        for i in range(len(self.weights) - 2,-1,-1):

            delta_hidden = hidden_error * d_sigmoid(hidden_outputs[i + 1])

            gradient_hidden =delta_hidden * self.lr
            next_hidden_error = self.weights[i].T @ delta_hidden
            self.bias[i] = self.bias[i] + gradient_hidden
            self.weights[i] = self.weights[i] + (gradient_hidden @ hidden_outputs[i].T)
            hidden_error = next_hidden_error

    def train(self, input, target):
        inputs = np.array(input).reshape(-1,1)
        target = np.array(target).reshape(-1,1)

        res = self.prediction(inputs)
        y = res[0]
        hidden_outputs = res[1]




        # backpropagation
        output_error = target - y
        delta_output = output_error * d_sigmoid(y)
        # output
        hidden_error = self.weights[-1].T @ delta_output

        gradient_output = delta_output * self.lr
        self.bias[-1] = self.bias[-1] + gradient_output
        self.weights[-1] = self.weights[-1] + (gradient_output @ hidden_outputs[-1].T)

        # hidden
        hidden_outputs.insert(0,inputs)

        self.backpropagation_hidden(hidden_outputs,hidden_error)

