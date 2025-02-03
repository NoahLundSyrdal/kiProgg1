import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
import numpy as np
from plants.bathtub_plant import BathtubPlant
from plants.cournot_plant import CournotPlant
from controllers.classic_controller import ClassicPIDController

class ControlSystem:
    def __init__(self, plant, controller, target, num_epochs=100):
        self.plant = plant
        self.controller = controller
        self.target = target
        self.num_epochs = num_epochs
        self.history = []
    
    def run_epochs(self, epochs=100):
        self.history = []
        for epoch in range(epochs):
          error = self.target - self.plant.current
          control_input = controller.update(error)
          current = plant.update(control_input)
          self.history.append(current)

    def get_history(self):
        return self.history
    
    def plot_results(self):
        heights = self.history
        times = np.linspace(0, self.num_epochs, self.num_epochs, endpoint=False)
        plt.plot(times, heights, label='Height')
        plt.axhline(self.target, color='r', linestyle='--', label='Setpoint')
        plt.xlabel('Time')
        plt.ylabel('Height')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    plant = BathtubPlant(current=1.0, area=10.0)
    controller = ClassicPIDController(k_p=2.0, k_i=0.1, k_d=0.01)
    system = ControlSystem(plant, controller, target=5)
    system.run_epochs()
    system.plot_results()

    plant = CournotPlant(10.0, 2.0, 0.4, 0.6)
    controller = ClassicPIDController(k_p=2.0, k_i=0.1, k_d=0.01)
    system = ControlSystem(plant, controller, target=5)
    system.run_epochs()
    system.plot_results()