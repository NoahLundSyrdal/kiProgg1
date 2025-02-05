import numpy as np
import jax
import jax.numpy as jnp


class NeuralNetworkController():
    def __init__(self, params):
        self.learning_rate = params["learning_rate"]
        self.num_layers = params["num_layers"]
        self.neurons_per_layer = params["neurons_per_layer"]
        self.weight_init_range = params["weight_init_range"]
        self.bias_init_range = params["bias_init_range"]
        self.activation_functions = self.get_activation_functions(
            params["activation_functions"])

    def init_params(self):
        layers = self.neurons_per_layer
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = np.random.normal(
                self.weight_init_range[0], self.weight_init_range[1], (int(sender), int(receiver)))
            biases = np.random.normal(
                self.bias_init_range[0], self.bias_init_range[1], (1, int(receiver)))
            params.append([weights, biases])
            sender = receiver
        return params

    def predict(self, params, features):
        activations = jnp.array(features)
        for (weights, biases), fn in zip(params, self.activation_functions):
            outputs = jnp.dot(activations, weights) + biases
            activations = fn(outputs)
        return activations

    def control_signal(self, params, error_history):
        P = jnp.asarray(error_history[-1]).reshape(())
        I = jnp.sum(jnp.array([jnp.asarray(err).reshape(())
                    for err in error_history]))
        D = jnp.asarray(error_history[-1] - error_history[-2]).reshape(())
        return self.predict(params, [P, I, D])

    def update_params(self, params, grads):
        updated_params = []
        for (w, b), (dw, db) in zip(params, grads):
            new_w = w - self.learning_rate * dw
            new_b = b - self.learning_rate * db
            updated_params.append((new_w, new_b))
        return updated_params

    def get_activation_functions(self, activation_function_names):
        activation_map = {
            "relu": lambda x: jnp.max(x, 0),
            "sigmoid": lambda x: 1/(1+jnp.exp(-x)),
            "tanh": lambda x: (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))
        }
        try:
            return [activation_map[func] for func in activation_function_names]
        except KeyError as e:
            raise ValueError(f"Invalid activation function: {e}")
