import jax.numpy as jnp
from jax import random

class LotkaVolterraPlant:
    def __init__(self, r, a, m, b, initial_prey, initial_predator, rng_key=None, carrying_capacity=None):
        self.r = r  # Prey growth rate
        self.a = a  # Predation rate
        self.m = m  # Predator mortality rate
        self.b = b  # Predator reproduction efficiency
        self.prey = initial_prey
        self.predator = initial_predator
        self.rng_key = rng_key
        self.carrying_capacity = carrying_capacity  # Carrying capacity for prey
    
    def update(self, dt=1, noise_scale=0.1):
        x, y = self.prey, self.predator
        
        # Logistic growth for prey with carrying capacity
        if self.carrying_capacity:
            dx = (self.r * (1 - x / self.carrying_capacity) - self.a * y) * x
        else:
            dx = (self.r - self.a * y) * x
        
        # Predator dynamics
        dy = (-self.m + self.b * x) * y
        
        # Apply independent noise
        self.rng_key, subkey_x, subkey_y = random.split(self.rng_key, 3)
        noise_dx = random.normal(subkey_x) * noise_scale
        noise_dy = random.normal(subkey_y) * noise_scale
        dx += noise_dx
        dy += noise_dy

        # Update populations with non-negative constraint
        self.prey = jnp.maximum(x + dx * dt, 0)
        self.predator = jnp.maximum(y + dy * dt, 0)

    def get_state(self):
        return self.prey, self.predator


        