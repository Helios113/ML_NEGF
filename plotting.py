import matplotlib.pyplot as plt
import pandas as pd

tar_dir = "test2_l"

test_dat = pd.read_csv("/home/staff/pa112h/Projects/ML_NEGF/res/{}/loss_test.txt".format(tar_dir))
train_dat = pd.read_csv("/home/staff/pa112h/Projects/ML_NEGF/res/{}/loss_train.txt".format(tar_dir))

cmpFile = open("/home/staff/pa112h/Projects/ML_NEGF/res/{}/info.txt".format(tar_dir))
cmp = cmpFile.readlines()[-1]
cmp = cmp.split(' ')[1]
cmpFile.close()

x_test = test_dat.iloc[:,3]
y_test = test_dat.iloc[:,2]

opt = x_test.argmin()

x_train = train_dat.iloc[:,3]
y_train = train_dat.iloc[:,2]



fig, ax = plt.subplots(1,1)
print(test_dat.iloc)

ax.plot(y_test,x_test,label="Test")
ax.plot(y_train,x_train,label="Train")

ax.axvline(opt,color="Red",label="opt",linestyle="dashed")
ax.axhline(cmp,color="Red",label="tar",linestyle="dashed")

ax.legend()
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss [MSE]")

plt.savefig('lines.png')