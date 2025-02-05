import jax.numpy as jnp
import numpy as np
import jax
from jax import random

class NeuralPIDController:
    def __init__(self, input_dim=3, hidden_layers=[5, 5], output_dim=1, activation_fn=jax.nn.relu, learning_rate=0.1, seed=42):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.key = random.PRNGKey(seed)
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate
    
    def forward(self, params, features):
        activations = jnp.array(features).reshape(1, -1)  # Ensure it's a row vector
        for (weights, biases) in params:
            outputs = jnp.dot(activations, weights) + biases  # Now the dimensions align correctly
            activations = self.activation_fn(outputs)

        return activations
    
    def update_params(self, params, gradients):
        updated_params = []
        for (w, b), (dw, db) in zip(params, gradients):
            updated_w = w - self.learning_rate * dw
            updated_b = b - self.learning_rate * db
            updated_params.append((updated_w, updated_b))

        return updated_params
    
    def update(self, params, error, i_error, d_error):
        return jnp.squeeze(self.forward(params, jnp.array([error, i_error, d_error])))


    def create_params(self):
        layers = [3, 5, 5, 1]
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = np.random.normal(
                -0.1, 0.1, (int(sender), int(receiver)))
            biases = np.random.normal(
                -0.05, 0.05, (1, int(receiver)))
            params.append([weights, biases])
            sender = receiver
        return params

