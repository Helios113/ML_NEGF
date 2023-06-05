import matplotlib.pyplot as plt
import pandas as pd

tar_dir = "test2_nobatch_03"
cut = 0 # ZERO FOR FULL GRAPH

# DATA COLLECTION
test_dat = pd.read_csv("/home/staff/pa112h/Projects/ML_NEGF/res/{}/loss_test.txt".format(tar_dir))
train_dat = pd.read_csv("/home/staff/pa112h/Projects/ML_NEGF/res/{}/loss_train.txt".format(tar_dir))

cmpFile = open("/home/staff/pa112h/Projects/ML_NEGF/res/{}/info.txt".format(tar_dir))
cmp = cmpFile.readlines()[-1]
cmp = round(float(cmp.split(' ')[1].replace('\n','')),6)
cmpFile.close()

y_test = test_dat.iloc[cut:,3]
x_test = test_dat.iloc[cut:,2]

y_train = train_dat.iloc[cut:,3]
x_train = train_dat.iloc[cut:,2]

opt = y_test.argmin() + cut


# PLOT GENERATION
fig, ax = plt.subplots(1,1)

ax.plot(x_test,y_test,label="Test")
ax.plot(x_train,y_train,label="Train")

ax.axvline(opt,color="Red",label="opt: {},{}".format(opt, round(y_test.min(),6)),linestyle="dashed")
ax.axhline(cmp,color="Red",label="tar: {}".format(cmp),linestyle="dashed")

ax.legend()
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss [MSE]")

plt.savefig("/home/staff/pa112h/Projects/ML_NEGF/res/{}/plot.png".format(tar_dir))