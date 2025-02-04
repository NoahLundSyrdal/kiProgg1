class ClassicPIDController:
    def __init__(self, k_p, k_i, k_d, learning_rate=0.1):  # Increased learning rate
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.prev_error = 0
        self.integral = 0
        self.learning_rate = learning_rate
        self.kp_history = []
        self.ki_history = []
        self.kd_history = []
    
    def update(self, error, dt=1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.k_p * error + self.k_i * self.integral + self.k_d * derivative
    
    def update_params(self, gradients):
        self.k_p -= self.learning_rate * gradients[0] * 5  # Scale updates
        self.k_i -= self.learning_rate * gradients[1] * 5
        self.k_d -= self.learning_rate * gradients[2] * 5
        
        self.kp_history.append(self.k_p)
        self.ki_history.append(self.k_i)
        self.kd_history.append(self.k_d)
        
        print(f"Updated Parameters - Kp: {self.k_p}, Ki: {self.k_i}, Kd: {self.k_d}")