import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
        