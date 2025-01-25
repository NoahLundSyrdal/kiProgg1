import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

class BathtubPlant:
    def __init__(self, height, area):
        self.height = height
        self.area = area
        self.cross_sectional_area = area / 100
        self.g = 9.8

    def step(self, U, dt = 1):
        subkey = random.split(jax.random.PRNGKey(0))[0]
        D = random.normal(subkey) * 0.1
        V = jnp.sqrt(2 * self.g * self.height)
        Q = self.cross_sectional_area * V
        db_dt = U + D - Q
        dh_dt = db_dt / self.area

        self.height += dh_dt * dt
        self.height = jnp.maximum(self.height, 0)
        return self.height, D
    


