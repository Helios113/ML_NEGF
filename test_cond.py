import numpy as np


dat = np.loadtxt("data_res/abs_pot/3x9/ml_test/NEGFXY_0.05_0.05_1_pot_1.txt")


print(dat.mean())
print(dat.std())
