import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class ResultsAnalysis:
    def __init__(self, simulation_batch, show_animation = False):
        """
        ResultsAnalysis class constructor
        """
        self.simulation_batch = simulation_batch
        
        self.first_frame = True

        if show_animation:
            self.animate()
        else:
            self.plot_XY()

    def plot_XY(self):
        """
        Plot the XY trajectory of the UGV
        """
        plt.figure()
        for sim in self.simulation_batch:
            plt.plot(sim.states[:, 0], sim.states[:, 1], label = sim.name_type)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('XY trajectory of the UGV')
        plt.legend()
        plt.show()

    def animate(self):
        """
        Animates the trajectory of the UGV.
        """

        self.animationSetup()

        anim = animation.FuncAnimation(self.fig, self.makeFrame, init_func=self.init_animation, frames=len(self.simulation_batch[0].states), interval=20, repeat=True) # blit=True

        # if self.path_planning.type == 'Regulation':
        #     self.axis.plot(self.path_planning.goal[0], self.path_planning.goal[1], 'r*', label='goal')
        self.axis.plot(self.simulation_batch[0].states[0, 0], self.simulation_batch[0].states[0, 1], 'bo', label='start')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('UGV Trajectory')
        plt.grid('minor')
        plt.legend()
        plt.show()

    def makeFrame(self, i):

        x_coord = []
        y_coord = []

        # todo: Add multiple simulations in animation. 
        x_coord.append(self.simulation_batch[0].states[:i, 0])
        y_coord.append(self.simulation_batch[0].states[:i, 1])

        # for sim in self.simulation_batch:
        #     x_coord.append(sim.states[:i, 0])
        #     y_coord.append(sim.states[:i, 1])
        #     break
        
        self.trajectory.set_data(x_coord, y_coord)
        return self.trajectory,

    def init_animation(self):
        self.trajectory.set_data([], [])
        return self.trajectory,    

    def animationSetup(self):
        self.fig = plt.figure()

        tol = 0.5 # tolerance for axis limits
        # todo: Add max/min based on inputs.
        maxs = np.max(self.simulation_batch[0].states, axis=0)
        mins = np.min(self.simulation_batch[0].states, axis=0)
        self.axis = plt.axes(xlim=(mins[0] - tol, maxs[0] + tol), ylim=(mins[1] - tol, maxs[1] + tol))
        self.trajectory, = self.axis.plot([], [], lw = 2)