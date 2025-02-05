import jax.numpy as jnp
import numpy as np

class CournotPlant:
    def __init__(self, p_max=50, cm=0.1, q1=0.5, q2=0.5, 
                 noise_range=(-0.01, 0.01)):
        self.p_max = p_max 
        self.cm = cm  
        self.noise_range = noise_range 
        self.initial_state = [5, q1, q2]
    
    def update(self, U, plant_state):
        D = np.random.uniform(self.noise_range[0], self.noise_range[1])
        q1 = jnp.clip(plant_state[1] + U, 0, 1)
        q2 = jnp.clip(plant_state[2] + D, 0, 1)

        q_total = q1 + q2
        price = self.p_max - q_total

        new_value = plant_state[1] * (price - self.cm)
        
        return [new_value, q1, q2]