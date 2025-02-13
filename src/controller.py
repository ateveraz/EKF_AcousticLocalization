from src.ugv.src.controller import Controller

import numpy as np

class PDController(Controller):
    def __init__(self):
        """
        PDController class constructor
        """
        super().__init__()
        self.kp = 8
        self.kd = 0.2
        self.previous_error = 0

    def compute_control(self, state, desired):
        """
        Computes the control input based on the current state.

        @param state: numpy vector containing the current state of the UGV
        @return: numpy vector containing the computed control input
        """
        error = desired - state[:2]
        distance_error = np.linalg.norm(error)
        angle_to_goal = np.arctan2(error[1], error[0])
        angle_error = angle_to_goal - state[2]

        # Normalize the angle error
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        error_vector = np.array([distance_error, angle_error])
        derivative = error_vector - self.previous_error
        self.previous_error = error_vector

        control = self.kp * error_vector + self.kd * derivative
        return control, error