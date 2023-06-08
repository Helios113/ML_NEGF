import matplotlib.pyplot as plt
import pandas as pd
import os 

subdirs = [x[0] for x in os.walk('paa1') if x[0] != 'paa1']

for dir in subdirs:
    try:
        tar_dir = dir
        cut = 0 # ZERO FOR FULL GRAPH

        # DATA COLLECTION
        test_dat = pd.read_csv("/home/staff/pa112h/Projects/ML_NEGF/{}/loss_test.txt".format(tar_dir))
        train_dat = pd.read_csv("/home/staff/pa112h/Projects/ML_NEGF/{}/loss_train.txt".format(tar_dir))

        cmpFile = open("/home/staff/pa112h/Projects/ML_NEGF/{}/info.txt".format(tar_dir))
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

        plt.savefig("/home/staff/pa112h/Projects/ML_NEGF/{}/plot.png".format(tar_dir))
    except Exception as e:
        print("failed to make plot at {} due to {}".format(tar_dir,e))