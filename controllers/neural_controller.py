import jax.numpy as jnp
from jax import grad
from jax import random

class NeuralController:
    def __init__(self, input_size, hidden_layers, output_size, rng_key, activation=jnp.tanh):
        self.weights = []
        self.biases = []
        self.activation = activation

        # Initialize weights and biases for all layers
        # Input layer to the first hidden layer
        rng_key, subkey = random.split(rng_key)
        self.weights.append(random.normal(subkey, shape=(input_size, hidden_layers[0])))
        self.biases.append(random.normal(subkey, shape=(hidden_layers[0],)))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            rng_key, subkey = random.split(rng_key)
            self.weights.append(random.normal(subkey, shape=(hidden_layers[i], hidden_layers[i + 1])))
            self.biases.append(random.normal(subkey, shape=(hidden_layers[i + 1],)))

        # Hidden layer to output layer
        rng_key, subkey = random.split(rng_key)
        self.weights.append(random.normal(subkey, shape=(hidden_layers[-1], output_size)))
        self.biases.append(random.normal(subkey, shape=(output_size,)))

    def forward(self, x):
        # Ensure input x is a 2D array with shape (1, input_size)
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)  # Add a batch dimension

        # Perform forward pass through the network
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(jnp.dot(x, w) + b)
        return jnp.dot(x, self.weights[-1]) + self.biases[-1]

    def update_weights(self, grads, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads["weights"][i]
            self.biases[i] -= learning_rate * grads["biases"][i]

    def compute_gradients(self, error):
        loss = lambda params: jnp.mean(jnp.square(error))
        return grad(loss)(self.get_params())

    def get_params(self):
        return {"weights": self.weights, "biases": self.biases}
