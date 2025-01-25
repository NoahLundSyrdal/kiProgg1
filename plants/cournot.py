import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class CournotPlant:
    def __init__(self, p_max, cm):
        self.q1 = 0.9
        self.q2 = 0.1
        self.p_max = p_max
        self.cm = cm

    def update(self, U, D):
        self.q1 = max(0, min(1, self.q1 + U))
        self.q2 = max(0, min(1, self.q2 + D))
        q_total = self.q1 + self.q2
        price = self.p_max - q_total
        profit = self.q1 * (price - self.cm)
        return profit

        