import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

"""
class CournotPlant:
    def __init__(self, p_max, c_m, T, dt=1):
        self.p_max = p_max
        self.c_m = c_m
        self.T = T
        self.dt = dt

        self.q1 = 0.5
        self.q2 = 0.5

    def update(self, U, D):
        self.q1 += U * self.dt
        self.q2 += D * self.dt

        q = self.q1 + self.q2
        p = self.p_max - q

        P1 = self.q1 * (p - self.c_m)

        E = self.T - P1
        
        return P1, E, p

    def reset(self, q1=0.5, q2=0.5):
        self.q1 = q1
        self.q2 = q2
"""
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

        