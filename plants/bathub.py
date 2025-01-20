import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

Height = input("Enter the height of the bathtub: ")
Area = input("Enter the area of the bathtub: ")
CrossSectionalArea = Area/100
Velocity = jnp.sqrt(2 * 9.8 * Height)
FlowRate = CrossSectionalArea * Velocity

U = input("Enter the output of the controller: ")
D = input("Enter the disturbance: ")

VolumeChange = U + D - FlowRate
WaterHeight = VolumeChange / CrossSectionalArea

