from src.ugv.src.ugv import UGV
from src.ugv.src.controller import Controller
from src.ugv.src.simulator import Simulator
from src.ugv.src.path_planning import PathPlanning

from filterpy.stats import plot_covariance_ellipse
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from src.acoustic import CMusic
from src.acoustic import Localization
import numpy as np
import matplotlib.pyplot as plt

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
        #error = desired - state
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

class SourceLocalizationPlanning(PathPlanning):
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

class AcousticLocalization(Simulator):
    def __init__(self, ugv, controller, path_planning, cmusic, localization, filter = None):
        """
        AcousticLocalization class constructor
        """
        super().__init__(ugv, controller, path_planning)
        self.cmusic = cmusic
        self.localization = localization

        if filter is not None:
            self.useFilter = True
            self.filter = filter
        else:
            self.useFilter = False

    def runCustom(self, time, show_animation = False):
        """
        Run simulation of the acoustic localization using a UGV. 
        """
        num_steps = int(time / self.ugv.params['dt'])

        self.states = np.zeros((num_steps, 3))
        self.errors = np.zeros((num_steps, 2))
        self.t = np.zeros(num_steps)

        self.states[0] = self.ugv.state

        self.first = True

        self.estimated_positions = np.zeros((num_steps, 2))

        if self.useFilter:
            self.filtered_position = np.zeros((num_steps, 2))
        
        dt = self.ugv.params['dt']

        for i in range(num_steps):
            self.t[i] = i * dt

            # Acoustic localization
            measured_angle = self.cmusic.measure(self.ugv.state[:2])
            estimated_position = self.localization.localize(self.ugv.state, measured_angle)
            self.estimated_positions[i] = estimated_position
            #print('Estimated position:', estimated_position, 'at time:', self.t[i])

            # Filtering if it is activated
            if self.useFilter:
                self.filter.updateRobotState(self.ugv.state)
                if self.first:
                    #print(np.shape(estimated_position))
                    #print( self.filter.hx(self.ugv.state[:2]) )
                    self.filter.set_initial_state(estimated_position)
                    self.first = False
                self.filter.predict()
                kalman_measure = np.array([estimated_position[0], estimated_position[1]])
                self.filter.update(kalman_measure)  #self.ugv.state[:2])
                self.filter.predict()
                self.filtered_position[i] = self.filter.x
                
                # Log filter output each 20 steps
                if i % 100 == 0:
                    print('Filtered position:', self.filtered_position[i], 'at time:', self.t[i])
                    print('Estimated position:', estimated_position, 'at time:', self.t[i])
                    # Check P matrix
                    print('det(P):', np.linalg.det(self.filter.P))
                    print('\n')


                # Use the filtered position when it is stable based on the filter covariance matrix. 
                if (np.linalg.det(self.filter.P) < 1e-4):
                    # Update the measured angle based on the filtered position
                    measured_angle = self.computeAngle(self.ugv.state[:2], self.filtered_position[i])
                    print('Angle updating based on filtered position:', self.filtered_position[i], 'at time:', self.t[i])

            # Path planning
            desired = self.path_planning.desired(self.ugv.state, measured_angle)
            control_input, error = self.controller.compute_control(self.ugv.state, desired)
            self.errors[i] = error
            self.ugv.update_state(control_input)
            self.states[i] = self.ugv.state

        if show_animation:
            self.animate()
        else:
            #self.plot()
            self.plotXY()
            if self.useFilter:
                self.plotFilter()
                # Mean of the last 10 estimated positions by filter
                print('Mean of the last 10 estimated positions by filter:', np.mean(self.filtered_position[-10:], axis = 0))
            print('Mean of the last 10 estimated positions:', np.mean(self.estimated_positions[-10:], axis = 0))

        return self.t, self.states

    def plotFilter(self):
        """
        Plot the filtered position of the source. 
        """
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.estimated_positions[:, 0], label = 'estimated x')
        plt.plot(self.t, self.filtered_position[:, 0], 'k:',label = 'Filtered x')
        plt.ylim(-50, 50)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.t, self.estimated_positions[:, 1], label = 'estimated y')
        plt.plot(self.t, self.filtered_position[:, 1], 'k:', label = 'Filtered y')
        plt.ylim(-50, 50)
        plt.legend()

        plt.show()

    def computeAngle(self, robot, source):
        """
        Compute the angle of arrival of the acoustic signal. 
        """
        return np.arctan2(source[1] - robot[1], source[0] - robot[0])

class CustomUKF(UnscentedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, process_noise, measurement_noise, initial_state):
        """
        CustomUKF class constructor
        """
        points = MerweScaledSigmaPoints(n = dim_x, alpha = 0.1, beta = 2., kappa = 1.) # kappa = 0 in the other code !
        super().__init__(dim_x = dim_x, dim_z = dim_z, dt = dt, hx = self.hx, fx = self.fx, points = points)

        self.x = initial_state
        self.P = np.eye(dim_x) * 100 # Check this value !
        self.R = measurement_noise * np.eye(dim_z)
        self.Q = process_noise * np.eye(dim_x)
        self.robot = np.array([0, 0, 0])
    
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
    
# Instance CMusic
source = np.array([-25, 30])
cmusic = CMusic(source, noise = 25 * (np.pi/180))

# Instance UGV
params = { 'dt': 0.01 }
initial_state = np.array([-1, 1, 0])
ugv = UGV(params, initial_state)

# Instance Controller
ctrl = PDController()

# Instance Path Planning
path_planning = SourceLocalizationPlanning(step_size = 0.5)

# Localization algorithm
loc = Localization(initial_state, 0)
loc = Localization(initial_state, 0)

# Instance UKF
measurement_noise = 10 #10 * (np.pi/180)
process_noise = 0
dim_x = 2
dim_z = 2

ukf = CustomUKF(dim_x, dim_z, params['dt'], process_noise, measurement_noise, np.array([0, 0]))

# Instance Simulator
sim = AcousticLocalization(ugv, ctrl, path_planning, cmusic, loc, filter = ukf)
sim.runCustom(20, show_animation = True)