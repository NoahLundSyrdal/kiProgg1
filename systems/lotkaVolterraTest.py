import jax.numpy as jnp
from jax import random
from lotka_volterra import LotkaVolterraPlant
from controllers.classic_controller import ClassicController

# Simulation setup
r = 0.1  # Initial prey growth rate
a = 0.02  # Predation rate
m = 0.1  # Predator mortality rate
b = 0.01  # Predator reproduction efficiency
initial_prey = 40.0
initial_predator = 9.0
target_prey = 50.0  # Target prey population

# Initialize plant and controller
rng_key = random.PRNGKey(42)
plant = LotkaVolterraPlant(r, a, m, b, initial_prey, initial_predator, rng_key)
controller = ClassicController(k_p=0.1, k_i=0.01, k_d=0.01)

# Simulation parameters
timesteps = 100
dt = 1.0

# Storage for plotting
prey_populations = []
predator_populations = []
control_outputs = []

# Run the simulation
for t in range(timesteps):
    prey, predator = plant.get_state()
    
    # Calculate control signal based on prey population error
    error = target_prey - prey
    control_output = controller.update(error, dt)
    
    # Apply control signal to prey growth rate
    plant.r = r + control_output
    
    # Step the plant
    plant.step(dt=dt)
    
    # Store data
    prey_populations.append(prey)
    predator_populations.append(predator)
    control_outputs.append(control_output)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Prey and predator populations
plt.subplot(2, 1, 1)
plt.plot(prey_populations, label="Prey Population")
plt.plot(predator_populations, label="Predator Population")
plt.axhline(y=target_prey, color='r', linestyle='--', label="Target Prey Population")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Lotka-Volterra Plant with Controller")
plt.legend()

# Control signal
plt.subplot(2, 1, 2)
plt.plot(control_outputs, label="Control Output")
plt.xlabel("Time")
plt.ylabel("Control Signal (Delta r)")
plt.legend()

plt.tight_layout()
plt.show()
