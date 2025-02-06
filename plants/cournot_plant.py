import jax.numpy as jnp
import numpy.random as rnd

class CournotPlant():
    def __init__(self, params):
        self.initial_state = params
        self.target_profit = params["target_profit"]
        self.pmax = params["max_price"]
        self.cm = params["marginal_cost"]
        self.noise = params["noise"]

    def update(self, control_signal, plant_state):
        q1 = jnp.clip(plant_state["q1"] + control_signal, 0, 1)
        D = rnd.uniform(self.noise[0], self.noise[1])
        q2 = jnp.clip(plant_state["q2"] + D, 0, 1)

        q_total = q1 + q2
        price = max(self.pmax - q_total, 0)

        profit_p1 = q1 * (price - self.cm)
        return {"profit_p1": profit_p1, "q1": q1, "q2": q2, "price": price}

    def get_error(self, state):
        """Return the difference between target profit and actual profit"""
        return state["profit_p1"] - self.target_profit
