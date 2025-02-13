from filterpy.stats import plot_covariance_ellipse
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

import numpy as np

class CustomUKF(UnscentedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, process_noise, measurement_noise, initial_state):
        """
        CustomUKF class constructor
        """
        points = MerweScaledSigmaPoints(n = dim_x, alpha = 0.1, beta = 2., kappa = 1.) 
        super().__init__(dim_x = dim_x, dim_z = dim_z, dt = dt, hx = self.hx, fx = self.fx, points = points)

        self.x = initial_state
        self.P = np.eye(dim_x) * 100 # Check this value !
        self.R = measurement_noise * np.eye(dim_z)
        self.Q = process_noise * np.eye(dim_x)
        self.robot = np.array([0, 0, 0])
        self.name_type = 'UKF'
    
    def set_initial_state(self, initial_state):
        """
        Set the initial state of the UKF
        """
        self.x = initial_state

    def hx(self, x):
        """
        Measurement function

        @param x: numpy vector containing the estimated speaker position. 
        """
        dx = self.robot[0] - x[0]
        dy = self.robot[1] - x[1]

        distance = np.sqrt(dx**2 + dy**2)
        abs_angle = np.arctan2(dy, dx)
        angle = np.arctan2(np.sin(abs_angle - self.robot[2]), np.cos(abs_angle - self.robot[2]))

        return np.array([distance, abs_angle])
        
    def fx(self, x, dt):
        """
        State transition function. 
        Assumption: the speaker's position is static. 
        """
        A = np.array([[1, 0], [0, 1]], dtype=float)
        return A @ x

    def updateRobotState(self, robot):
        """
        Update the robot position
        """
        self.robot = robot