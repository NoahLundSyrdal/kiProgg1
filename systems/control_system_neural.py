import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
import numpy as np
from plants.bathtub_plant import BathtubPlant
from plants.cournot_plant import CournotPlant
from controllers.neural_controller import NeuralPIDController

class ControlSystem:
    def __init__(self, plant, controller, target, num_epochs=20, num_timesteps=50):
        self.plant = plant
        self.controller = controller
        self.target = target
        self.num_epochs = num_epochs
        self.num_timesteps = num_timesteps
        self.history = []
    
    def run_epoch(self):
        error_history = jnp.zeros(self.num_timesteps)
        for t in range(self.num_timesteps):
            error = self.target - self.plant.current
            error_history = error_history.at[t].set(error)
            integral_error = jnp.sum(error_history[:t+1])
            derivative_error = (error - error_history[t-1]) if t > 0 else 0.0
            control_signal = float(self.controller.control(jnp.array([error, integral_error, derivative_error])).squeeze())
            self.plant.update(control_signal)
        self.history.append(float(self.plant.current))

        return float(jnp.mean(error_history ** 2))  # Mean Squared Error
    

    def train(self):
        for epoch in range(self.num_epochs):
            mse = self.run_epoch()

            # Fix: Error terms need to be computed correctly
            error = self.target - self.plant.current
            error_terms = jnp.array([error, 0.0, 0.0])  # Ignore derivative/integral terms for now

            self.controller.train_step(error_terms, jnp.array(mse))  # Ensure inputs to JAX are arrays
            print(f"Epoch {epoch+1}/{self.num_epochs}, MSE: {mse:.4f}")
    


    def get_history(self):
        return self.history
    
    def plot_results(self):
        heights = self.history
        times = np.linspace(0, self.num_epochs, self.num_epochs, endpoint=False)
        plt.figure(figsize=(10, 20))
        
        plt.plot(times, heights, label='Learning Porgression')
        plt.axhline(self.target, color='r', linestyle='--', label='Target')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    plant = BathtubPlant(area=100, current=10)
    controller = NeuralPIDController()
    system = ControlSystem(plant, controller, target=5)
    system.train()
    system.plot_results() 
    
    plant = CournotPlant(10.0, 2.0, 0.4, 0.6)
    controller = NeuralPIDController(activation_fn=jax.nn.tanh, learning_rate=0.01)
    system = ControlSystem(plant, controller, target=5.0)
    system.train()
    system.plot_results() 