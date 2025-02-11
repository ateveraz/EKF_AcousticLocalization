import numpy as np

class CMusic:
    def __init__(self, source, noise = 0.1):
        """
        Contrained MUSIC algorithm constructor. 

        This class emulates the Constrained MUSIC algorithm for acoustic source localization.
        """
        self.real_source = source
        self.angular_noise = noise

    def measure(self, robot_position):
        """ 
        Measures the angle of arrival of the acoustic signal.

        @param robot_position: numpy vector containing the position of the robot.
        @return: angle of arrival of the signal.
        """

        # Compute the angle of arrival of the signal
        direction = self.real_source - robot_position
        angle = np.arctan2(direction[1], direction[0]) + np.random.normal(0, self.angular_noise)
        return angle