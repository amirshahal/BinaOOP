import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.w = np.random.rand(output_size, input_size)
        self.b = np.zeros((output_size, 1))
        self.z = None
        self.a = None
        self.x = None
        self.dz = None
        self.dw = None
        self.db = None

    def forward_propagation(self, x):
        self.x = x
        self.z = np.dot(self.w, x) + self.b
        self.a = self.activation_function(self.z)

    def back_propagation(self, last_output, last_weights, m):
        if last_weights is None:
            self.dz = self.a - last_output
        else:
            self.dz = np.dot(last_weights.T, last_output) * self.a * (1 - self.a)
        self.dw = np.dot(self.dz, self.x.T) / m
        self.db = np.sum(self.dz, axis=1, keepdims=True)

    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db


class NN:
    def __init__(self, verbose=False):
        self.layers = []
        self.lr = 0.1
        self.iterations = 10000
        self.losses = []
        self.verbose = verbose

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, x, y):
        for i in range(self.iterations):
            input_for_next_layer = x
            for layer_index, layer in enumerate(self.layers):
                layer.forward_propagation(input_for_next_layer)
                input_for_next_layer = layer.a

            output = self.layers[-1].a
            m = y.shape[0]
            loss = -(1 / m) * np.sum(y * np.log(output) + (1 - y) * np.log(1 - output))
            if self.verbose and (i < 10 or i % 1000 == 0):
                print(f"Iteration {i}, loss= {loss}")
            self.losses.append(loss)

            last_output = y
            last_weights = None
            for layer in reversed(self.layers):
                layer.back_propagation(last_output, last_weights, m)
                last_output = layer.dz
                last_weights = np.copy(layer.w)
                layer.update(self.lr)

        if self.verbose:
            for i, layer in enumerate(self.layers):
                print(f"w{i+1}= {layer.w}")

    def test(self, x):
        input_for_next_layer = x
        for layer in self.layers:
            layer.forward_propagation(input_for_next_layer)
            input_for_next_layer = layer.a

        return input_for_next_layer
