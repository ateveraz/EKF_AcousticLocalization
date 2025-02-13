from src.ugv.src.simulator import Simulator
from src.acoustic import CMusic, Localization
from src.controller import PDController


from src.pathPlanning import Planning
from src.ugv.src.ugv import UGV


import numpy as np
import matplotlib.pyplot as plt

class AcousticLocalization(Simulator):
    def __init__(self, params, filter = None, debugging = False):
        """
        AcousticLocalization class constructor
        """
        self.ugv = UGV(params, params['initial_state'])
        self.controller = PDController()
        self.path_planning = Planning(params['step_size'])

        super().__init__(self.ugv, self.controller, self.path_planning)

        self.cmusic = CMusic(params['source'], noise = params['noise'])
        self.localization = Localization(params['initial_state'])

        self.debugging = debugging

        if filter is not None:
            self.useFilter = True
            self.filter = filter
            self.name_type = filter.name_type
        else:
            self.useFilter = False
            self.name_type = 'Not filtered'


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

            # Filtering if it is activated
            if self.useFilter:
                self.filter.updateRobotState(self.ugv.state)
                if self.first:
                    self.filter.set_initial_state(estimated_position)
                    self.first = False
                self.filter.predict()
                kalman_measure = np.array([estimated_position[0], estimated_position[1]])
                self.filter.update(kalman_measure)  #self.ugv.state[:2])
                self.filter.predict()
                self.filtered_position[i] = self.filter.x
                
                # Log filter output each 20 steps
                if i % 100 == 0 and self.debugging:
                    print('Filtered position:', self.filtered_position[i], 'at time:', self.t[i])
                    print('Estimated position:', estimated_position, 'at time:', self.t[i])
                    # Check P matrix
                    print('det(P):', np.linalg.det(self.filter.P))
                    print('\n')


                # Use the filtered position when it is stable based on the filter covariance matrix. 
                if (np.linalg.det(self.filter.P) < 1e-4):
                    # Update the measured angle based on the filtered position
                    measured_angle = self.computeAngle(self.ugv.state[:2], self.filtered_position[i])
                    if self.debugging:
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