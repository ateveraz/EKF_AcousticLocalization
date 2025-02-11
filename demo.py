from src.ugv.src.ugv import UGV
from src.acoustic import CMusic
import numpy as np


source = np.array([10, 10])
cmusic = CMusic(source)
robot = np.array([0, 0])
print(cmusic.measure(robot))