import matplotlib.pyplot as plt

import matplotlib.colors as colors
import numpy as np
my_data = np.genfromtxt('test.csv', delimiter=',', skip_header=True)


fig, ax = plt.subplots(figsize=(6,12))
test_data = my_data[:,0]
data = test_data.reshape(71,26, order='F')
pos = ax.imshow(data, cmap='viridis', interpolation='bilinear')
c_bar = fig.colorbar(pos, ax=ax)


ax.set_xlabel("X [nm]")
ax.set_ylabel("Y [nm]")
c_bar.set_label("e")
plt.tight_layout()
plt.show()
