import numpy as np
import jax
import jax.numpy as jnp


class NeuralNetworkController():
    def __init__(self, params):
        self.learning_rate = params["learning_rate"]
        self.num_layers = params["num_layers"]
        self.neurons_per_layer = params["neurons_per_layer"]
        self.weights_range = params["weights_range"]
        self.biases_range = params["biases_range"]
        self.activation_function = self.get_activation_function(
            params["activation_function"])

    def init_params(self):
        layers = self.neurons_per_layer
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = np.random.normal(
                self.weights_range[0], self.weights_range[1], (int(sender), int(receiver)))
            biases = np.random.normal(
                self.biases_range[0], self.biases_range[1], (1, int(receiver)))
            params.append([weights, biases])
            sender = receiver
        return params

    def predict(self, params, features):
        activations = jnp.array(features)
        for (weights, biases) in params:
            outputs = jnp.dot(activations, weights) + biases
            activations = self.activation_function(outputs)
        return activations
    
    def control_signal(self, params, error_history):
        kp = error_history[-1]
        ki = sum(error_history)
        kd = error_history[-1] - error_history[-2]
        return self.predict(params, jnp.asarray([kp, ki, kd]).reshape((1,3))) 

    def update_params(self, params, gradients):
        updated_params = []
        for (w, b), (dw, db) in zip(params, gradients):
            new_w = w - self.learning_rate * dw
            new_b = b - self.learning_rate * db
            updated_params.append((new_w, new_b))
        return updated_params

    def get_activation_function(self, activation_function):
        match activation_function:
            case "relu":
                return jax.nn.relu
            case "sigmoid":
                return jax.nn.sigmoid
            case "tanh":
                return jax.nn.tanh
            case _:
                raise ValueError(f"Invalid: {activation_function}")