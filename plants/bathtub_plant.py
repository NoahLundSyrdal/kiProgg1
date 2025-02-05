import jax.numpy as jnp
import numpy.random as rnd

class BathtubPlant():
    def __init__(self, params):
        """Initialize the bathtub model with parameters."""
        self.initial_state = params
        self.gravity = 9.81
        self.target_height = params["target_height"]
        self.area = params["area"]
        self.cross_sectional_area = params["cross_sectional_area"]
        self.noise = params["noise"]

    def update(self, control_signal, plant_state):
        h = plant_state["water_height"] 
        V = jnp.sqrt(2 * self.gravity * max(h, 0))  
        Q = V *  self.cross_sectional_area
        D = rnd.uniform(self.noise[0], self.noise[1])
        db_dt = control_signal + D - Q
        dh_dt = db_dt / self.area
 
        return {"water_height": max(0, h + dh_dt)}

    def get_error(self, state):
        return state["water_height"] - self.target_height