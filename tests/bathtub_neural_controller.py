import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("plants"))
sys.path.append(os.path.abspath("controllers")) 
from jax import random
from plants.bathtub import BathtubPlant
from controllers.neural_controller import NeuralController
import numpy as np

# Initialize plant and controller
rng_key = random.PRNGKey(0)
plant = BathtubPlant(height=1.0, area=10.0, rng_key=rng_key)
controller = NeuralController(input_size=3, hidden_layers=[5], output_size=1, rng_key=rng_key)

# Simulation parameters
setpoint = 5.0
dt = 1.0
num_steps = 100

# Storage for plotting
heights = []
times = []

error_history = []
integral_error = 0
prev_error = 0
# Simulation loop
for t in range(num_steps):
    error = setpoint - plant.height
    integral_error += error * dt
    derivative_error = (error - prev_error) / dt
    prev_error = error
    mse = np.mean(np.square(error))
    grads = controller.compute_gradients(mse)
    controller.update_weights(grads, 0.01)
    height = plant.step(controller.forward(np.array([error, integral_error, derivative_error])), dt)
    
    heights.append(height)
    times.append(t * dt)

# Plot results
plt.plot(times, heights, label='Height')
plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint')
plt.xlabel('Time')
plt.ylabel('Height')
plt.legend()
plt.show()

