import jax.numpy as jnp
import jax
from jax import grad, jit, random
import optax

class NeuralPIDController:
    def __init__(self, input_dim=3, hidden_layers=[32, 32], output_dim=1, activation_fn=jax.nn.relu, learning_rate=0.03, seed=42):
        self.key = random.PRNGKey(seed)
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate

        self.params = []
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(layer_sizes) - 1):
            key, self.key = random.split(self.key)
            w = random.normal(key, (layer_sizes[i], layer_sizes[i+1])) * 0.1
            b = jnp.zeros(layer_sizes[i+1])
            self.params.append((w, b))

        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)
    
    def forward(self, x, params):
        for i, (w, b) in enumerate(params[:-1]):
            x = self.activation_fn(jnp.dot(x, w) + b)
        w_out, b_out = params[-1]
        return jnp.dot(x, w_out) + b_out
    
    def loss_fn(self, params, x, target):
        pred = self.forward(x, params)
        return jnp.mean((pred - target) ** 2)
        
    def control(self, error_terms):
        x = jnp.array(error_terms)
        return self.forward(x, self.params)
    
    def train_step(self, error_terms, target):
        loss, grads = jax.value_and_grad(self.loss_fn)(self.params, error_terms, target)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss