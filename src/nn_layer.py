import numpy as np



def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return np.where(x > 0, 1.0, 0.0)
def leaky_relu(x):
    return np.maximum(0.01*x, x)

def d_leaky_relu(x):
    return np.where(x > 0, 1.0, 0.01)

def linear(x):
    return x

def d_linear(x):
    return 1

def sigmoid(x):
    # Clip per evitare overflow in np.exp
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)



functions = {relu: drelu, linear: d_linear, sigmoid: d_sigmoid, leaky_relu: d_leaky_relu}

class layer:
    def __init__(self, row, col, activation_function):
        self.weights = np.random.uniform(-0.5,0.5, size=(row, col))
        self.func = activation_function
        self.dfunc = functions[self.func]
        self.bias = np.random.uniform(-0.5,0.5, size=(row,1))

    @property
    def W(self):
        return self.weights
    @W.setter
    def W(self, x):
        self.weights = x
    @property
    def b(self):
        return self.bias
    @b.setter
    def b(self, x):
        self.bias = x
    @property
    def activation_function(self):
        return self.func
    
    def __str__(self):
        return f"{self.weights}"