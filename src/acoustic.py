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

class Localization:
    def __init__(self, init_robot_position, init_measured_angle = 0):
        """
        Localization algorithm using triangulation. 
        """
        self.robot_position = init_robot_position
        self.measured_angle = init_measured_angle

    def localize(self, robot_position, measured_angle):
        """
        Localizes the source of the acoustic signal using triangulation.

        @param robot_position: numpy vector containing the position of the robot.
        @param measure: angle of arrival of the signal.
        @return: estimated position of the source.
        """
        # Compute the estimated position of the source
        At = np.array([[np.sin(self.measured_angle), -np.cos(self.measured_angle)], [np.sin(measured_angle), -np.cos(measured_angle)]]) 
        bt = np.array([self.robot_position[0]*np.sin(self.measured_angle) - self.robot_position[1]*np.cos(self.measured_angle), robot_position[0]*np.sin(measured_angle) - robot_position[1]*np.cos(measured_angle)])
        
        # Update the robot position and measured angle
        self.measured_angle = measured_angle
        self.robot_position = robot_position

        return np.linalg.solve(At, bt)
