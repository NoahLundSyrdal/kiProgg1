import jax
import jax.numpy as jnp
from jax import random

class BathtubPlant:
    def __init__(self, area, current):
        self.current = current
        self.area = area
        self.cross_sectional_area = area / 100
        self.g = 9.8

    def update(self, U, dt = 1):
        subkey = random.split(jax.random.PRNGKey(0))[0]
        D = random.normal(subkey) * 0.1
        V = jnp.sqrt(2 * self.g * self.current)
        Q = self.cross_sectional_area * V
        db_dt = U + D - Q
        dh_dt = db_dt / self.area

        self.current += dh_dt * dt
        self.current = jnp.maximum(self.current, 0)
        return self.current