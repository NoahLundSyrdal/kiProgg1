import matplotlib.pyplot as plt
from plants.cournot import CournotPlant
from controllers.classic_controller import ClassicController

plant = CournotPlant(p_max=10.0, c_m=2.0, T=5.0)
controller = ClassicController(k_p=2.0, k_i=0.1, k_d=0.01)

# Simulation parameters
setpoint = 5.0
dt = 1.0
num_steps = 100

# Storage for plotting
profits = []
times = []

# Initialize P1
P1 = 0

# Simulation loop
for t in range(num_steps):
    control_input = controller.update(setpoint - P1, dt)
    P1, E, p = plant.update(U=control_input, D=0.1)
    
    profits.append(P1)
    times.append(t * dt)

# Plot results
plt.plot(times, profits, label='Profit')
plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint')
plt.xlabel('Time')
plt.ylabel('Profit')
plt.legend()
plt.show()


