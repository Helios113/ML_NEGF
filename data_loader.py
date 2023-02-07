import matplotlib.pyplot as plt

import matplotlib.colors as colors
import numpy as np

my_data = np.loadtxt('data_3x3_10/ml_res/NEGF_0.33125_0.05_54_tar.txt')
print(my_data.shape)
fig, ax = plt.subplots(figsize=(6,12))
pos = ax.imshow(my_data, cmap='viridis', interpolation='bilinear')
c_bar = fig.colorbar(pos, ax=ax)


ax.set_xlabel("X [nm]")
ax.set_ylabel("Y [nm]")
c_bar.set_label("1/cm3")
# plt.tight_layout()
plt.show()
