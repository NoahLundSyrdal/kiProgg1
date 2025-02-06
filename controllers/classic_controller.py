import jax.numpy as jnp

class ClassicController():
    def __init__(self, params):
        self.learning_rate = params["learning_rate"]

    def update_params(self, params, gradients):
        gradients_norm = jnp.linalg.norm(gradients)

        if gradients_norm > 1:
            gradients = gradients / gradients_norm
            
        kp, ki, kd = params
        kp -= self.learning_rate * gradients[0]
        ki -= self.learning_rate * gradients[1]
        kd -= self.learning_rate * gradients[2]
        return jnp.asarray([kp, ki, kd])

    def control_signal(self, params, error_history):
        kp, ki, kd = params
        return kp*error_history[-1] + ki * jnp.sum(jnp.array(error_history)) + kd*(error_history[-1]-error_history[-2])