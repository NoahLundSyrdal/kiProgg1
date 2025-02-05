import jax.numpy as jnp

class ClassicController():
    def __init__(self, params):
        self.learning_rate = params["learning_rate"]

    def update_params(self, params, grads):
        max_norm = 1.0  # Set a max gradient norm
        grad_norm = jnp.linalg.norm(grads)

        # If gradients exceed max_norm, scale them down
        grads = jnp.where(grad_norm > max_norm, grads *
                          (max_norm / grad_norm), grads)

        return params - self.learning_rate * grads

    def control_signal(self, params, error_history):
        kp, ki, kd = params

        P = kp*error_history[-1]
        I = ki * jnp.sum(jnp.array(error_history))
        D = kd*(error_history[-1]-error_history[-2])

        return P+I+D

    def print_params(self, params):
        kp, ki, kd = params
        print("kp: " + kp + ", ki: " + ki + ", kd: " + kd)
