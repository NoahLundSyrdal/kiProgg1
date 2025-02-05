import jax.numpy as jnp
import numpy as np

class BathtubPlant:
    def __init__(self, initial_state, cross_sectional_area, area, noise_range=(-0.01, 0.01)):
        self.initial_state = initial_state
        self.cross_sectional_area = cross_sectional_area
        self.area = area
        self.g = 9.8
        self.noise_range = noise_range

    def update(self, U, plant_state):
        current_water_height = plant_state[0]
        D = np.random.uniform(self.noise_range[0], self.noise_range[1])
        V = jnp.sqrt(2 * self.g * current_water_height)
        Q = self.cross_sectional_area * V
        db_dt = U + D - Q
        dh_dt = db_dt / self.area

        current_water_height += dh_dt
        current_water_height = jnp.maximum(current_water_height, 0)
        return current_water_height