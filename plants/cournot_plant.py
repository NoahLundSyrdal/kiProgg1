import jax.numpy as jnp
import numpy as np

class CournotPlant:
    def __init__(self, p_max=1.5, cm=0.1, q1_init=0.5, q2_init=0.5, 
                 noise_range=(-0.01, 0.01)):
        self.p_max = p_max 
        self.cm = cm  
        self.q1 = q1_init 
        self.q2 = q2_init 
        self.noise_range = noise_range  
        self.current = 0  
    
    def update(self, U):
        self.q1 = jnp.clip(self.q1 + U, 0, 1)
        D = np.random.uniform(self.noise_range[0], self.noise_range[1])
        self.q2 = jnp.clip(self.q2 + D, 0, 1)

        q_total = self.q1 + self.q2

        price = self.p_max - q_total

        self.current = self.q1 * (price - self.cm)
        
        return self.current
 