import jax.numpy as jnp
from jax import grad

class NeuralPIDController:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = jnp.tanh(jnp.dot(x, w) + b)  
        return x

    def update_weights(self, loss_grad):
        self.weights -= 0.01 * loss_grad
