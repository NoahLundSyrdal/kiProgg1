from jax import random
from jax import value_and_grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from plants.bathtub_plant import BathtubPlant
from plants.cournot_plant import CournotPlant
from plants.lotka_volterra import LotkaVolterraPlant
from controllers.classic_controller import ClassicPIDController
from controllers.neural_controller import NeuralPIDController

class ControlSystem:
    def __init__(self, plant, controller, params, target, num_epochs=20, num_timesteps=200):
        self.plant = plant
        self.controller = controller
        self.target = target
        self.num_epochs = num_epochs
        self.num_timesteps = num_timesteps
        self.params_history = [params]
        self.mse_history = []
    
    def calculate_mse(self, error_history):
        return jnp.mean(jnp.square(jnp.array(error_history)))
        
    
    def run_epoch(self, params):
        error_history = []
        d_error = 0
        i_error = 0
        prev_error = 0
        plant_state = self.plant.initial_state
    
        for _ in range(self.num_timesteps):
            error = self.target - plant_state[0]
            d_error = error - prev_error
            i_error += error
            prev_error = error
            error_history.append(error)

            control_input = self.controller.update(params, error, i_error, d_error) 
            plant_state = self.plant.update(control_input, plant_state)

        return self.calculate_mse(error_history) 
    
    def run_epochs(self):
        for epoch in range(self.num_epochs):
            params = self.params_history[-1]
            mse, gradients = value_and_grad(self.run_epoch)(params)
            self.mse_history.append(np.abs(mse))
            new_params = self.controller.update_params(params, gradients)
            self.params_history.append(new_params)
            print(f'Epoch: {epoch+1}/{self.num_epochs} - MSE: {round(float(np.abs(mse)), 4)}')
   
    def plot_results(self):
        x = np.array(range(len(self.mse_history)))  # Convert range to NumPy array
        y = self.mse_history
        plt.plot(x, y)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Learning Progression")
        plt.show()

        if isinstance(self.controller, ClassicPIDController):
            x = np.array(range(len(self.params_history)))
            y1, y2, y3 = zip(*self.params_history)
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.plot(x, y3)
            plt.legend(["k_p", "k_i", "k_d"])
            plt.title("Control Parameters")
            plt.show()

if __name__ == "__main__":
    """
    plant = BathtubPlant([110], 4, 3)
    params = [0.5, 0.5, 0.5]
    controller = ClassicPIDController()
    system = ControlSystem(plant, controller, params, target=100)
    system.run_epochs()
    system.plot_results()
    """

    plant = CournotPlant(noise_range=(-0.001, 0.001))
    controller = NeuralPIDController()
    params = controller.create_params()
    system = ControlSystem(plant, controller, params, target=6)
    system.run_epochs()
    system.plot_results() 
    
"""     plant = BathtubPlant([110], 4, 3)
    params = [0.5, 0.5, 0.5]
    
    controller = NeuralPIDController()
    params = controller.create_params()
    system = ControlSystem(plant, controller, params, target=100)
    system.run_epochs()
    system.plot_results()  """