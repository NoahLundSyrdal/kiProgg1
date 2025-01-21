import matplotlib.pyplot as plt
from jax import random
from plants.bathtub import BathtubPlant
from controllers.classic_controller import ClassicController

# Initialize plant and controller
rng_key = random.PRNGKey(0)
plant = BathtubPlant(height=1.0, area=10.0, rng_key=rng_key)
controller = ClassicController(k_p=2.0, k_i=0.1, k_d=0.01)

# Simulation parameters
setpoint = 5.0
dt = 1.0
num_steps = 100

# Storage for plotting
heights = []
times = []

# Simulation loop
for t in range(num_steps):
    error = setpoint - plant.height
    control_input = controller.update(error, dt)
    height, disturbance = plant.step(control_input, dt)
    
    heights.append(height)
    times.append(t * dt)

# Plot results
plt.plot(times, heights, label='Height')
plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint')
plt.xlabel('Time')
plt.ylabel('Height')
plt.legend()
plt.show()

