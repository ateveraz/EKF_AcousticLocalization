import matplotlib.pyplot as plt
import numpy as np

class ResultsAnalysis:
    def __init__(self, simulation_batch):
        """
        ResultsAnalysis class constructor
        """
        self.simulation_batch = simulation_batch

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