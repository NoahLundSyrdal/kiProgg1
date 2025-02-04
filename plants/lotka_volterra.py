import jax
import jax.numpy as jnp
import numpy as np
from jax import random

class LotkaVolterraPlant:
    def __init__(self, r=0.5, a=0.1, m=0.1, b=0.02, initial_prey=10.0, initial_predator=5.0, 
                 noise_range=(-0.1, 0.1), carrying_capacity=None):
        self.r = r  # Prey growth rate
        self.a = a  # Predation rate
        self.m = m  # Predator mortality rate
        self.b = b  # Predator reproduction efficiency
        self.prey = initial_prey
        self.predator = initial_predator
        self.noise_range = noise_range  # Noise range for randomness
        self.carrying_capacity = carrying_capacity  # Carrying capacity for prey
        self.current = 0  # Placeholder for output

    def update(self, U):
        # Apply user input to prey population (e.g., resource changes)
        self.prey = jnp.maximum(self.prey + U, 0)
        
        # Introduce noise to predator population
        D = np.random.uniform(self.noise_range[0], self.noise_range[1])
        self.predator = jnp.maximum(self.predator + D, 0)
        
        # Compute total interaction effects
        if self.carrying_capacity:
            prey_growth = (self.r * (1 - self.prey / self.carrying_capacity) - self.a * self.predator) * self.prey
        else:
            prey_growth = (self.r - self.a * self.predator) * self.prey
        
        predator_growth = (-self.m + self.b * self.prey) * self.predator
        
        # Update populations
        self.prey = jnp.maximum(self.prey + prey_growth, 0)
        self.predator = jnp.maximum(self.predator + predator_growth, 0)
        
        # Define a measure of system state (e.g., predator-prey balance)
        self.current = self.prey - self.predator  # Example metric
        
        return self.current


        