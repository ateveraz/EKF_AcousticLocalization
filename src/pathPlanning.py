from src.ugv.src.path_planning import PathPlanning

import numpy as np

class Planning(PathPlanning):
    def __init__(self, step_size):
        """
        Path planning class constructor for source localization. 
        """
        super().__init__('Quasi-tracking')
        self.step_size = step_size
    
    def desired(self, robot_state, angle):
        """
        Computes the desired state of the UGV at a given time step.

        @param robot_state: numpy vector containing the current state of the UGV.
        @param angle: angle of arrival of the acoustic signal.
        @return: numpy vector containing the desired state of the UGV.
        """
        return robot_state[:2] + self.step_size * np.array([np.cos(angle) + np.sin(angle), np.sin(angle) - np.cos(angle)])