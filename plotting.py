import matplotlib.pyplot as plt
import pandas as pd

dat = pd.read_csv("results/test1/loss_test.txt")


fig, ax = plt.subplots(1,1)
print(dat.iloc)

ax.plot(dat.iloc[:,2],dat.iloc[:,3])
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss [MSE]")

plt.show()