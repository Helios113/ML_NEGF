import numpy as np
from os.path import join
import os
import argparse
import pandas as pd
#import KYLES CODE

class Aggregator:
    def __init__(self, args):
        self.root = args.root
        self.target = args.target

        self.dataframe = pd.DataFrame() # [plane, vg, vd, location, type, iter, filedir, shape, index]

        self.avgmappings = {'avg_cur':0,'abs_pot':1}
        self.planemappings = {'xy':0,'yz':1,'zx':2}
        self.typemappings = {'charge':0,'pot':1}

    def read(self):
        #getting all files from all subdirectories and paths to said files
        imgs = []
        name_as_lst = []
        VGnums = set()
        VDnums = set()
        target_prefix = "NEGFXY"

        filenum = 0
        filenames = []
        for root, dirs, files in os.walk("/home/bailey/ML_NEGF/main_data_dir"):
            for name in files:
                if name.startswith(target_prefix):
                    filenames.append(os.path.join(root, name))
                    name = str(name).split(".txt")[0] #Removing .txt at end of strings
                    name_as_lst = name.split("_")
                    XY = name_as_lst.pop(0)[-2:]
                    name_as_lst.insert(0,XY)
                    imgs.append(name_as_lst + str(root).split("/")[-3:])

        df = pd.DataFrame(imgs,columns=['Plane','VG','VD','Location',"Type","Iteration","Criteria","Shape","Index"])
        self.dataframe = df

    def find_max_tier(self, df, row):
        df.query("Plane == '{}' and VG == '{}' and VD == '{}' and Location == '{} and Type == '{}' and Citeria == '{} and Shape == '{}'".format(
            row["Plane"], row["VG"], row["VD"], row["Location"], row["Type"], row["Citeria"], row["Shape"]))
        
        return df.max("Iteration")

    def condition(self, df):
        # Condition should go through the entire self.array and condition based on the iteration number and data type
        # Get all entries with iterator == 1


        norm = [[0,0],[0,0]]
        unique_devices = df.query("Iterator == '1'")

        for index, row in df.iterrows():
            if row["Type"] == "Charge":
                data = np.log10(np.loadtxt(row["Index"]))
                chrg = 1
            else:
                data = np.loadtxt(row["Index"])
                chrg = 0

            norm[chrg][0] = data.mean()
            norm[chrg][1] = data.std()
            


arguments = argparse.ArgumentParser()
arguments.add_argument('--root', help="Root directory of the data on file")
arguments.add_argument('-t','--target', type=str, help="Target directory for coditioned data", default='tar_dir/')

aggregator = Aggregator(arguments)