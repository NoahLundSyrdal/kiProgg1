from matplotlib import pyplot as plt
from plants.bathtub_plant import BathtubPlant
from plants.cournot_plant import CournotPlant
from plants.lotka_volterra_plant import LotkaVolterraPlant
from controllers.classic_controller import ClassicController
from controllers.neural_network_controller import NeuralNetworkController

import json
import jax
import jax.numpy as jnp
import numpy as np

class ControlSystem:
    def __init__(self, plant, controller, num_epochs, num_timesteps, params):
        self.plant = plant
        self.controller = controller
        self.num_epochs = num_epochs
        self.num_timesteps = num_timesteps
        self.grad_fn = jax.value_and_grad(self.run_epoch)
        self.params_history = [params]
        self.mse_history = []

    def run_epochs(self):
        for _ in range(self.num_epochs):
            mse, gradients = self.grad_fn(self.params_history[-1])
            self.params_history.append(self.controller.update_params(self.params_history[-1], gradients))
            self.mse_history.append(mse)
            print(f"Epoch: {len(self.mse_history)}, MSE: {mse}")

    def run_epoch(self, params):
        error_history = [0, 0]
        plant_state = self.plant.initial_state
        for _ in range(self.num_timesteps):
            control_signal = self.controller.control_signal(params, error_history)
            plant_state = self.plant.update(control_signal, plant_state)
            error = self.plant.get_error(plant_state)
            error_history.append(error)

        error_history = [jnp.asarray(err).reshape(()) for err in error_history]
        return jnp.mean(jnp.square(jnp.array(error_history)))

    
    def plot_results(self):
        x = np.array(range(len(self.mse_history))) 
        y = self.mse_history
        plt.plot(x, y)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Learning Progression")
        plt.show()

        if isinstance(self.controller, ClassicController):
            x = np.array(range(len(self.params_history)))
            y1, y2, y3 = zip(*self.params_history)
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.plot(x, y3)
            plt.legend(["k_p", "k_i", "k_d"])
            plt.title("Control Parameters")
            plt.show()

if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)  

    match (config["general"]["plant"]):
        case "bathtub_plant":
            plant = BathtubPlant(config["bathtub_plant"])
        case "cournot_plant":
            plant = CournotPlant(config["cournot_plant"])
        case "lotka_volterra_plant":
            plant = LotkaVolterraPlant(config["lotka_volterra_plant"])
        case _:
            raise ValueError(f"Unknown plant: {config["general"]["plant"]}")

    match (config["general"]["controller"]):
        case "classic_controller":
            controller = ClassicController(config["classic_controller"])
            params = jnp.array([config["classic_controller"]["kp"], config["classic_controller"]["ki"], config["classic_controller"]["kd"]])
        case "neural_network_controller":
            controller = NeuralNetworkController(config["neural_network_controller"])
            params = controller.init_params()
        case _:
            raise ValueError(f"Unknown controller: {config["general"]["controller"]}")

    consys = ControlSystem(plant, controller, num_epochs=config["general"]["num_epochs"],
                        num_timesteps=config["general"]["num_timesteps"], params=params)
    consys.run_epochs()
    consys.plot_results()