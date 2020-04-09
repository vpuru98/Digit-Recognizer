
import numpy as np


# Defining Activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(x * -1))

def tanh(x):
    return (np.exp(x) - np.exp(x * -1)) / (np.exp(x) + np.exp(x * -1))

def relu(x):
    return np.maximum(x, 0)

def dSigmoid(x):
    y = sigmoid(x)
    return y * (1 - y)

def dTanh(x):
    y = tanh(x)
    return 1 - y ** 2

def dRelu(x):
    return np.array(x >= 0, dtype='float64')

activation_function_dictionary = {
    'sigmoid': {
        'function': sigmoid,
        'derivative': dSigmoid
    },
    'relu': {
        'function': relu,
        'derivative': dRelu
    },
    'tanh': {
        'function': tanh,
        'derivative': dTanh
    }
}


# Defining Initializers

def zero_initialization(dim1, dim2):
    return np.zeros((dim1, dim2))

def random_initializtion(dim1, dim2):
    np.random.seed(0)
    return np.random.randn(dim1, dim2)

def xavier_initialization(units, prev_units):
    np.random.seed(0)
    return np.random.randn(units, prev_units) * ((1 / prev_units) ** 0.5)

def he_initialization(units, prev_units):
    np.random.seed(0)
    return np.random.randn(units, prev_units) * ((2 / prev_units) ** 0.5)

initialization_dictionary = {
    'zero': zero_initialization,
    'random': random_initializtion,
    'xavier': xavier_initialization,
    'he': he_initialization
}


# Defining Cost Functions

def mse(outputs, labels):
     m = np.size(outputs, axis=1)
     cost = np.sum((outputs - labels) ** 2) / (2 * m)
     return cost

def mse_derivative(outputs, labels):
    m = np.size(outputs, axis=1)
    outputs = outputs + 10 ** -12
    return (outputs - labels) / m

def cross_entropy(outputs, labels):
    m = np.size(outputs, axis=1)
    cost = -np.sum((labels * np.log(outputs + 10 ** -12)) + ((1 - labels) * 
                   np.log((1 - outputs) + 10 ** -12))) 
    return cost / m

def cross_entropy_derivative(outputs, labels):
    m = np.size(outputs, axis=1)
    outputs = outputs + 10 ** -12
    return ((1 - labels) / (1 - outputs) - labels / outputs) / m

def regularization(regularization_constant, layers):
    m = np.size(layers[0].parameters['W'], axis=1)
    regularization_cost = 0
    for layer in layers:
        regularization_cost += (np.sum(layer.parameters['W'] ** 2) + 
                                    np.sum(layer.parameters['b'] ** 2)) 
    return regularization_cost * (regularization_constant / (2 * m))

def regularization_derivative(regularization_constant, layers):
    m = np.size(layers[0].parameters['W'], axis=1)
    return [
            {
                'W': layer.parameters['W'] * (regularization_constant / m), 
                'b': layer.parameters['b'] * (regularization_constant / m)
            }
            for layer in layers
    ]
    
cost_functions_dictionary = {
    'mse': {
        'function': mse,
        'derivative': mse_derivative
    },
    'cross_entropy': {
        'function': cross_entropy,
        'derivative': cross_entropy_derivative
    },
    'regularization': {
        'function': regularization,
        'derivative': regularization_derivative
    }
}


# Defining Layer class

class Layer:

    def __init__(self, layer_type):
        self.layer_type = layer_type
    
    def forward_propagate(self, X, predict=False):
        pass
    
    def backward_propagate(self, dA, X):
        pass


class Dense(Layer):

    def __init__(self, activation, units, prev_units, initializer='xavier'):
        super().__init__('Dense')
        assert activation in activation_function_dictionary
        assert initializer in initialization_dictionary
        self.function = activation_function_dictionary[activation]['function']
        self.derivative = activation_function_dictionary[activation]['derivative']
        self.parameters = {
            'W': initialization_dictionary[initializer](units, prev_units),
            'b': initialization_dictionary['zero'](units, 1)
        }

    def forward_propagate(self, X, predict=False):
        self.pre_activations = np.dot(self.parameters['W'], X) + self.parameters['b']
        self.activations = self.function(self.pre_activations)
        return self.activations
    
    def backward_propagate(self, dA, X):
        dZ = dA * self.derivative(self.pre_activations)
        dW = np.dot(dZ, X.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(self.parameters['W'].T, dZ)
        return dW, db, dA


class BatchNorm(Layer):
    
    def __init__(self, prev_units):
        super().__init__('BatchNorm')
        self.units = prev_units
        self.parameters = {
            'W': np.ones((self.units, 1)),
            'b': np.zeros((self.units, 1))
        }
        self.batch_count = 0

    def forward_propagate(self, X, predict=False):
        if(self.batch_count == 0):
            self.avg_mean = np.zeros((X.shape[0], 1))
            self.avg_dev = np.zeros((X.shape[0], 1))
        if(not predict):
            self.batch_mean = np.mean(X, axis=1, keepdims=True)
            self.batch_dev = (np.var(X, axis=1, keepdims=True) + 10 ** -12) ** 0.5
            self.avg_mean += self.batch_mean
            self.avg_dev += self.batch_dev
            self.batch_count += 1
            self.activations = (self.parameters['W'] * ((X - self.batch_mean) / 
                                        self.batch_dev) + self.parameters['b'])
        else:
            self.activations = (self.parameters['W'] * ((X - (self.avg_mean / 
                                    self.batch_count)) / (self.avg_dev / 
                                    self.batch_count)) + self.parameters['b'])
        return self.activations

    def backward_propagate(self, dA, X):
        m = np.size(X, axis=1)
        dW = np.sum(dA * ((X - self.batch_mean) / self.batch_dev), axis=1, keepdims=True)
        db = np.sum(dA, axis=1, keepdims=True)
        dA = dA * (self.parameters['W'] * (((m - 1) / m) * (1 - (1 / m) * (((X - 
                        self.batch_mean) / self.batch_dev) ** 2)) *  (1 / self.batch_dev)))
        return dW, db, dA


# Defining ANN class

class ANN:

    @staticmethod
    def shuffle(X, Y):
        X_shuffled = X.copy()
        Y_shuffled = Y.copy()
        for i in range(np.size(X_shuffled, axis=1)):
            np.random.seed(0)
            swap_index = np.random.randint(i, np.size(X_shuffled, axis=1))
            X_temp = X_shuffled[:, i]
            Y_temp = Y_shuffled[:, i]
            X_shuffled[:, i] = X_shuffled[:, swap_index]
            Y_shuffled[:, i] = Y_shuffled[:, swap_index]
            X_shuffled[:, swap_index] = X_temp
            Y_shuffled[:, swap_index] = Y_temp
        return X_shuffled, Y_shuffled


    @staticmethod
    def get_mini_batch_splits(mini_batch_size, dataset_size):
        splits = []
        index = 0
        while index < dataset_size:
            splits.append((index, min(index + mini_batch_size - 1, dataset_size - 1)))
            index += mini_batch_size
        return splits


    def __init__(self, input_size, cost_function):
        assert cost_function in cost_functions_dictionary
        self.cost_function = cost_functions_dictionary[cost_function]
        self.last_input_size = input_size
        self.layers = []


    def add_layer_Dense(self, activation, units, initializer='xavier'):
        self.layers.append(Dense(activation, units, self.last_input_size, initializer))
        self.last_input_size = units
    

    def add_layer_BatchNorm(self):
        self.layers.append(BatchNorm(self.last_input_size))


    def descend_with_batch(self, X_batch, Y_batch, learning_rate, beta1, beta2, 
        regularization_constant, V_W, V_b, S_W, S_b):
        outputs = X_batch
        for j in range(len(self.layers)):
            outputs = self.layers[j].forward_propagate(outputs)

        regularization_derivatives = (cost_functions_dictionary['regularization']['derivative']
                                      (regularization_constant, self.layers))
        dA = self.cost_function['derivative'](outputs, Y_batch)
        for j in range(-1, -len(self.layers) - 1, -1):
            dW, db, dA = self.layers[j].backward_propagate(dA, X_batch if 
                j == -len(self.layers) else self.layers[j - 1].activations)
            dW = dW + regularization_derivatives[j]['W']
            db = db + regularization_derivatives[j]['b']
            
            V_W[j]['updates'] += 1
            V_W[j]['values'] = beta1 * V_W[j]['values'] + dW
            V_W[j]['values'] = V_W[j]['values'] * ((1 - beta1) / (1 - beta1 ** (V_W[j]['updates'])))
            V_b[j]['updates'] += 1
            V_b[j]['values'] = beta1 * V_b[j]['values'] + db
            V_b[j]['values'] = V_b[j]['values'] * ((1 - beta1) / (1 - beta1 ** (V_b[j]['updates'])))
            S_W[j]['updates'] += 1
            S_W[j]['values'] = beta2 * S_W[j]['values'] + dW ** 2
            S_W[j]['values'] = S_W[j]['values'] * ((1 - beta2) / (1 - beta2 ** (S_W[j]['updates'])))
            S_b[j]['updates'] += 1
            S_b[j]['values'] = beta2 * S_b[j]['values'] + db ** 2
            S_b[j]['values'] = S_b[j]['values'] * ((1 - beta2) / (1 - beta2 ** (S_b[j]['updates'])))

            self.layers[j].parameters['W'] = (self.layers[j].parameters['W'] - learning_rate * 
                            (V_W[j]['values'] / (S_W[j]['values'] ** 0.5 + 10 ** -12)))
            self.layers[j].parameters['b'] = (self.layers[j].parameters['b'] - learning_rate * 
                            (V_b[j]['values'] / (S_b[j]['values'] ** 0.5 + 10 ** -12)))

            self.layers[j].parameters['W'] += 10 ** -12
            self.layers[j].parameters['b'] += 10 ** -12


    def fit(self, X, Y, num_iterations=100, learning_rate=0.001, beta1=0.9, 
        beta2=0.999, regularization_constant=0, mini_batch_size = 512):
        assert self.layers[0].parameters['W'].shape[1] == X.shape[0]
        X, Y = self.shuffle(X ,Y)
        X = self.normalize(X)

        V_W = [{'values': np.zeros(layer.parameters['W'].shape), 'updates': 0} 
               for layer in self.layers]
        V_b = [{'values': np.zeros(layer.parameters['b'].shape), 'updates': 0} 
               for layer in self.layers]
        S_W = [{'values': np.zeros(layer.parameters['W'].shape), 'updates': 0} 
               for layer in self.layers]
        S_b = [{'values': np.zeros(layer.parameters['b'].shape), 'updates': 0} 
               for layer in self.layers]
        
        if mini_batch_size == -1:
            mini_batch_size = np.size(X, axis=1)
        splits = self.get_mini_batch_splits(mini_batch_size, np.size(X, axis=1))
        costs = []
        for i in range(num_iterations + 1):
            for split in splits:
                X_batch = X[:, split[0]:split[1] + 1]
                Y_batch = Y[:, split[0]:split[1] + 1]
                self.descend_with_batch(X_batch, Y_batch, learning_rate, beta1, 
                            beta2, regularization_constant, V_W, V_b, S_W, S_b)

            if (i + 1) % 5 == 0 or i + 1 == 1:
                outputs = X
                for j in range(len(self.layers)):
                    outputs = self.layers[j].forward_propagate(outputs)
                cost = round(self.cost_function['function'](outputs, Y) + 
                            cost_functions_dictionary['regularization']['function']
                            (regularization_constant, self.layers), 4)
                print('Cost after epoch #{} = {}'.format(i + 1, cost))
                costs.append(cost)

        plt.plot([i + 1 for i in range(len(costs))], costs, color='black')
        plt.title('Cost Variation')
        plt.ylabel('Costs')
        plt.show()
    

    def normalize(self, X):
        self.means = np.mean(X, axis=1)
        self.scales = np.amax(X, axis=1) - np.amin(X, axis=1)
        for i in range(len(self.scales)):
            if self.scales[i] == 0:
                self.scales[i] = 1
        self.means = self.means.reshape((np.size(X, axis=0), 1))
        self.scales = self.scales.reshape((np.size(X, axis=0), 1))
        X = (X - self.means) / self.scales
        return X
    

    def predict(self, X):
        X = (X - self.means) / self.scales
        outputs = X
        for i in range(len(self.layers)):
            outputs = self.layers[i].forward_propagate(outputs, predict=True)
        return outputs



