from src.simulator import AcousticLocalization
from src.results_analysis import ResultsAnalysis as results
from src.filter import CustomUKF

import numpy as np
import matplotlib.pyplot as plt

# Dictionary with the customized parameters for each simulations. 
# The keys are the names of the simulations and the values are dictionaries with the parameters of the simulations.
params = { 'dt': 0.01 , 'noise': 25 * (np.pi/180), 'initial_state': np.array([-1, 1, 0]), 'source': np.array([-25, 30]), 'measurement_noise': 10, 'process_noise': 0, 'dim_x': 2, 'dim_z': 2, 'step_size': 0.5 }

ukf = CustomUKF(params['dim_x'], params['dim_z'], params['dt'], params['process_noise'], params['measurement_noise'], params['initial_state'])

# Instance Simulator 1
sim1 = AcousticLocalization(params, debugging = False)
sim1.runCustom(20, show_animation = False)

# Instance Simulator 2
sim2 = AcousticLocalization(params, filter = ukf, debugging = False)
sim2.runCustom(20, show_animation = False)

# Instance ResultsAnalysis
data = results([sim1, sim2])
data.plot_XY()