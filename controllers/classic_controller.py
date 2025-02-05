class ClassicPIDController:
    def __init__(self, learning_rate=0.001): 
        self.learning_rate = learning_rate

    def update(self, params, error, i_error, d_error, dt=1):
        kp, ki, kd = params[0], params[1], params[2]
        return kp * error + ki * i_error + kd * d_error
    
    def update_params(self, params, gradients):
        kp = params[0] * (1 - self.learning_rate * gradients[0])
        ki = params[1] * (1 - self.learning_rate * gradients[1])
        kd = params[2] * (1 - self.learning_rate * gradients[2])

        return (kp, ki, kd)