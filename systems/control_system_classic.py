import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
import numpy as np
from jax import grad
from plants.bathtub_plant import BathtubPlant
from plants.cournot_plant import CournotPlant
from plants.lotka_volterra import LotkaVolterraPlant
from controllers.classic_controller import ClassicPIDController

class ControlSystem:
    def __init__(self, plant, controller, target, num_epochs=100):
        self.plant = plant
        self.controller = controller
        self.target = target
        self.num_epochs = num_epochs
        self.history = []
        self.mse_history = []
    
    def calculate_mse(self, error_history):
        return jnp.mean(jnp.square(jnp.array(error_history)))
    
    def run_epochs(self):
        self.history = []
        error_history = []
        
        # Ensure parameter histories are reset
        self.controller.kp_history = []
        self.controller.ki_history = []
        self.controller.kd_history = []
        
        def loss_fn(params):
            kp, ki, kd = params
            temp_controller = ClassicPIDController(kp, ki, kd)

            if isinstance(self.plant, BathtubPlant):
                temp_plant = BathtubPlant(area=self.plant.area, current=self.plant.current)
            elif isinstance(self.plant, CournotPlant):
                temp_plant = CournotPlant(self.plant.market_size, self.plant.num_firms, self.plant.costs, self.plant.elasticity)
            elif isinstance(self.plant, LotkaVolterraPlant):
                temp_plant = LotkaVolterraPlant()

            temp_error_history = []

            for _ in range(self.num_epochs):
                error = self.target - temp_plant.current
                control_input = temp_controller.update(error)
                temp_plant.update(control_input)
                temp_error_history.append(error)
            return self.calculate_mse(temp_error_history)
        
        for epoch in range(self.num_epochs):
            error = self.target - self.plant.current
            control_input = self.controller.update(error)
            self.plant.update(control_input)
            self.history.append(self.plant.current)
            error_history.append(error)
            
            mse = self.calculate_mse(error_history)
            self.mse_history.append(mse)
            gradients = grad(loss_fn)((self.controller.k_p, self.controller.k_i, self.controller.k_d))
            self.controller.update_params(gradients)
            
            # Store PID parameter values after update
            self.controller.kp_history.append(self.controller.k_p)
            self.controller.ki_history.append(self.controller.k_i)
            self.controller.kd_history.append(self.controller.k_d)
    
    def plot_results(self):
        times = np.linspace(0, self.num_epochs, self.num_epochs, endpoint=False)
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, self.history, label='Response')
        plt.axhline(self.target, color='r', linestyle='--', label='Setpoint')
        plt.xlabel('Time')
        plt.ylabel('System Response')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, self.controller.kp_history, label='Kp')
        plt.plot(times, self.controller.ki_history, label='Ki')
        plt.plot(times, self.controller.kd_history, label='Kd')
        plt.xlabel('Epoch')
        plt.ylabel('Parameter Value')
        plt.legend()
        plt.title('PID Parameter Updates')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, self.mse_history, label='MSE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.title('MSE over Epochs')
        plt.show()

if __name__ == "__main__":
    plant = BathtubPlant(current=1.0, area=10.0)
    controller = ClassicPIDController(k_p=2.0, k_i=0.1, k_d=0.01, learning_rate=0.1)
    system = ControlSystem(plant, controller, target=5)
    system.run_epochs()
    system.plot_results()
    
    plant = CournotPlant(10.0, 2.0, 0.4, 0.6)
    controller = ClassicPIDController(k_p=2.0, k_i=0.1, k_d=0.01, learning_rate=0.1)
    system = ControlSystem(plant, controller, target=5)
    system.run_epochs()
    system.plot_results()
    
    plant = LotkaVolterraPlant()
    controller = ClassicPIDController(k_p=2.0, k_i=0.1, k_d=0.01, learning_rate=0.1)
    system = ControlSystem(plant, controller, target=50)
    system.run_epochs()
    system.plot_results()
