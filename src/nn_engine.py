import numpy as np
from nn_layer import *

class neural_network:
    def __init__(self, n_hidden_layer, size_hidden, size_input, size_output, learning_rate, hidden_function, output_function):
        self.size = n_hidden_layer + 1 #1 per l'output layer sempre presente
        self.output_size = size_output
        self.lr = learning_rate

        
        self.hiddens_layers = []
        # il layer di input consiste nel primo hidden layer

        # creo i layer intermedi (hidden) + input layer
        if n_hidden_layer > 0:
            self.input_layer = layer(size_hidden, size_input, hidden_function)
            self.hiddens_layers.append(self.input_layer)
            if n_hidden_layer - 1 > 0:
                for _ in range(n_hidden_layer - 1):
                    self.hiddens_layers.append(layer(size_hidden, size_hidden, hidden_function))
            
        self.output_layer = layer(size_output, size_hidden if n_hidden_layer > 0 else size_input, output_function)
        
        # necessari per la backpropagation
        self.valori_intermedi = []
        self.valori_input = []

    def feedforward(self, x):
        self.valori_input = []
        self.valori_intermedi = []
        
        x = np.array(x).reshape(-1,1)
        # input layer
        # t = self.input_layer.W @ x + self.input_layer.b
        # self.valori_intermedi.append(t)
        # t = self.input_layer.activation_function(t)


        # print(f"{self.input_layer.W} \n x \n{x} \n + \n {self.input_layer.b} \n = \n {self.input_layer.W @ x + self.input_layer.b}\n\nsigma(z_i) = \n {t}")
        
        
        # hidden layer

        t = x

        for layer in self.hiddens_layers:
            self.valori_input.append(t)
            t = layer.W @ t + layer.b
            self.valori_intermedi.append(t)
            # print(t)
            t = layer.activation_function(t)

        self.valori_input.append(t)
        # print(F"valori intermedi: \n {self.valori_intermedi}")
        # print(F"valori input: \n {self.valori_input}")

        # output layer

        
        t = self.output_layer.W @ t + self.output_layer.b
        
        self.valori_intermedi.append(t)
        res = self.output_layer.func(t)

        return res
    
    def feedback(self, input, target):
        target = np.array(target).reshape(-1,1)
        guess = self.feedforward(input)
        deltas = []
        
        
        # delta dell'output
        delta_out = self.dMSE(target, guess) * self.output_layer.dfunc(self.valori_intermedi[-1])
        deltas.append(delta_out)


        # calcolo i delta degli hidden layer, l'ultimo deve utilizzare la matrice dell'output layer

        # I layer precedenti utilizzano i pesi del layer alla loro destra

        current_delta = delta_out
        next_W = self.output_layer.W
        

        for i in range(len(self.hiddens_layers) - 1, -1, -1):
            current_delta = (next_W.T @ current_delta) * self.hiddens_layers[i].dfunc(self.valori_intermedi[i])

            deltas.append(current_delta)

            next_W = self.hiddens_layers[i].W

        # AGGIORNAMENTO DEI PARAMETRI
        
        # PESI e BIAS
        # in modo che facciamo corrispondere gli indici dei delta
        # Se prima era [output, hidden_N, ..., hidden_0]
        # Ora diventa [hidden_0, hidden_1, ..., output]
        deltas_rev = list(reversed(deltas))
        
        for i in range(len(self.hiddens_layers)):
            self.hiddens_layers[i].W -= self.lr * (deltas_rev[i] @ self.valori_input[i].T)
            self.hiddens_layers[i].b -= self.lr * deltas_rev[i]

        self.output_layer.W -= self.lr * (deltas_rev[-1] @ self.valori_input[-1].T)
        self.output_layer.b -= self.lr * deltas_rev[-1]
    

    def MSE(self,target, guess):
        somma = 0
        for i in range(0,self.output_size):
            somma += (target[i] - guess[i])**2
        return somma/2
    def dMSE(self,target, guess):
        return guess - target

    

    def __str__(self):
        testo = "-"* 35 + "\n" +f"Caratteristiche della rete neurale:\n#layer: {self.size:<10}\n"
        i = 0
        for x in self.hiddens_layers:
            testo += f"{i}. neuroni: {x.W.size:<10}\n"
            i+=1
        testo += f"output. neuroni: {self.output_layer.W.size:<10}\n" + "-"*35+ "\n"
        return testo
        
        
        


        











