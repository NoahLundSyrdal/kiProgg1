import numpy as np
import numpy.random as rnd

class LotkaVolterraPlant:
    def __init__(self, params):
        self.initial_state = params
        self.prey_birth_rate = params["prey_birth_rate"]
        self.predation_rate = params["predation_rate"]
        self.predator_reproduction_rate = params["predator_reproduction_rate"]
        self.predator_death_rate = params["predator_death_rate"]
        self.noise = params["noise"]
        self.target_prey = params["target_prey"]
        
    def update(self, control_signal, plant_state):

        D = rnd.uniform(self.noise[0], self.noise[1])
 
        prey = plant_state["prey"]
        predator = plant_state["predator"]
        
        prey_growth = self.prey_birth_rate * prey - self.predation_rate * prey * predator
        predator_growth = self.predator_reproduction_rate * prey * predator - self.predator_death_rate * predator
        
        prey += (prey_growth + control_signal) 
        predator += (predator_growth + D) 

        prey = max(prey, 0)
        predator = max(predator, 0)
        
        return {"prey": prey, "predator": predator}

    def get_error(self, state):
        return state["prey"] - self.target_prey
