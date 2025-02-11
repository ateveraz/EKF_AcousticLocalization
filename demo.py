from src.ugv.src.ugv import UGV
from src.ugv.src.controller import Controller
from src.ugv.src.simulator import Simulator
from src.ugv.src.path_planning import PathPlanning

from src.acoustic import CMusic
from src.acoustic import Localization
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
    def __init__(self, ugv, controller, path_planning, cmusic, localization):
        """
        AcousticLocalization class constructor
        """
        super().__init__(ugv, controller, path_planning)
        self.cmusic = cmusic
        self.localization = localization

    def runCustom(self, time, show_animation = False):
        """
        Run simulation of the acoustic localization using a UGV. 
        """
        num_steps = int(time / self.ugv.params['dt'])

        self.states = np.zeros((num_steps, 3))
        self.errors = np.zeros((num_steps, 2))
        self.t = np.zeros(num_steps)

        self.states[0] = self.ugv.state
        
        for i in range(num_steps):
            # Acoustic localization
            measured_angle = self.cmusic.measure(self.ugv.state[:2])
            estimated_position = self.localization.localize(self.ugv.state, measured_angle)
            print('Estimated position:', estimated_position, 'at time:', self.t[i])

            # Path planning
            self.t[i] = i * self.ugv.params['dt']
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

        return self.t, self.states
        
    
# Instance CMusic
source = np.array([-10, 10])
cmusic = CMusic(source, noise = 1)

# Instance UGV
params = { 'dt': 0.01 }
initial_state = np.array([-1, 1, 0])
ugv = UGV(params, initial_state)

# Instance Controller
ctrl = PDController()

# Instance Path Planning
path_planning = SourceLocalizationPlanning(step_size = 5)

# Instance Simulator
sim = AcousticLocalization(ugv, ctrl, path_planning, cmusic, Localization(initial_state, 0))
sim.runCustom(20, show_animation = True)