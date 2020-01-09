
import numpy as np


class ANN:


    def __init__(self, X, y, layers=None):
        self.X = np.array(X, dtype='float64')
        self.y = np.array(y, dtype='float64')
        self.features = len(X[0])
        self.classes = len(y[0])
        self.layers = [self.features, self.classes]
        if layers is not None:
            self.layers.pop()
            self.layers.extend(layers)
            self.layers.append(self.classes)

        self.m = len(X)
        self.layer_count = len(self.layers)


    def init_parameters(self):
        self.parameters = []
        for i in range(self.layer_count - 1):
            np.random.seed(0)
            self.parameters.append(np.random.randn(self.layers[i + 1], 
                        self.layers[i] + 1) * np.sqrt(1 / self.layers[i]))


    def init_network(self):
        self.neurons = []
        for i in range(self.layer_count):
            self.neurons.append(np.zeros(self.layers[i]))


    def forwardpropagation(self, inp):
        self.neurons[0] = inp
        for i in range(1, self.layer_count):
            layer_prev = np.ones(self.layers[i - 1] + 1)
            layer_prev[1:] = self.neurons[i - 1]
            self.neurons[i] = np.reciprocal(np.exp((self.parameters[i - 1]
                                @ layer_prev) * -1) + 1)


    def backpropagation(self):
        self.gradients = []
        for i in range(self.layer_count - 1):
            self.gradients.append(np.zeros((self.layers[i + 1], 
                                                self.layers[i] + 1)))

        for i in range(self.m):
            if i % 5000 == 0:
                print('\tBackpropagation: Running for example number {}'.
                        format(i))
                
            self.forwardpropagation(self.X[i, :])
            delta = np.zeros((self.classes, 1))
            delta[:, 0] = self.neurons[self.layer_count - 1] - self.y[i]
            for j in range(self.layer_count - 1, 0, -1):
                layer_prev = np.ones((self.layers[j - 1] + 1, 1))
                layer_prev[1:, 0] = self.neurons[j - 1]
                self.gradients[j - 1] += delta @ layer_prev.transpose()
                delta_prev = (((self.parameters[j - 1].transpose() @ delta) * 
                               (layer_prev * (1 - layer_prev)))[1:, :])
                delta = delta_prev
    

    def cost_function(self):
        def cost_per_example(example_number):
            self.forwardpropagation(self.X[example_number, :])
            output = self.neurons[-1]
            cost = 0.0
            for j in range(self.classes):
                cost1, cost2 = 0, 0
                try:
                    cost1 = float((-self.y[example_number][j]) * 
                                  (np.log(output[j]))) / self.m
                except ValueError:
                    cost1 = -1000.0
                try:
                    cost2 = (float((1 - self.y[example_number][j]) * 
                                   (np.log(1 - output[j]))) / self.m) * -1
                except ValueError:
                    cost2 = -1000.0

                cost += cost1 + cost2

            return cost

        cost = 0.0
        for i in range(self.m):
            cost += cost_per_example(i)

        return cost


    def gradient_descent(self, alpha, n_iter):
        for i in range(n_iter):
            print('Gradient Descent: Running iteration {}'.format(i + 1))
            self.backpropagation()
            for j in range(self.layer_count - 1):
                self.parameters[j] -= (alpha / self.m) * self.gradients[j]
            
            if i % 30 == 0:
                print("Gradient Descent: Dumping object after iteration {}".
                      format(i + 1))


    def feature_scaling(self):
        self.means = np.mean(self.X, axis=0)
        self.scales = np.amax(self.X, axis=0) - np.amin(self.X, axis=0)
        for i in range(len(self.scales)):
            if self.scales[i] == 0:
                self.scales[i] = 1

        for i in range(self.m):
            self.X[i, :] = self.X[i, :] - self.means
            self.X[i, :] = self.X[i, :] / self.scales


    def fit(self, alpha=0.3, n_iter=500, resume=True):
        self.feature_scaling()
        self.init_network()
        self.init_parameters(resume)
        self.gradient_descent(alpha, n_iter)


    def predict(self, inp):
        x = np.array(inp, dtype='float64')
        x -= self.means
        x /= self.scales
        self.forwardpropagation(x)
        return self.neurons[self.layer_count - 1]
