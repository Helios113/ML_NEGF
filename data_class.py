import numpy as np
from os.path import join
import argparse
import pandas
#import KYLES CODE

class Aggregator:
    def __init__(self, args):
        self.root = args.root
        self.target = args.target

        self.array = np.array() # [plane, vg, vd, location, type, iter, filedir, shape, index]

        self.avgmappings = {'avg_cur':0,'abs_pot':1}
        self.planemappings = {'xy':0,'yz':1,'zx':2}
        self.typemappings = {'charge':0,'pot':1}

    def read(self):
        # Kyles code
        pass

    def query(self):
        # Querys the array
        pass

    def condition(self):
        # Condition should go through the entire self.array and condition based on the iteration number and data type
        # Get all entries with iterator == 1


        norm = [[0,0],[0,0]]
        unique_devices = df.query("Iterator == '1'")

        
        for index, row in df.iterrows():
            maxIter = find_max_iter(device)
            filepath = device.filepath
            
            if device.type == 1:
                data = np.log10(np.loadtxt(device.filepath))
            else:
                data = np.loadtxt(device.filepath)

            norm[device.type][0] = data.mean()
            norm[device.type][1] = data.std()

            np.savetxt(join(self.target,device.name),(data - norm[device.type][0])/(norm[device.type][1]))
    
            if device.type == 0:


arguments = argparse.ArgumentParser()
arguments.add_argument('--root', help="Root directory of the data on file")
arguments.add_argument('-t','--target', type=str, help="Target directory for coditioned data", default='tar_dir/')

aggregator = Aggregator(arguments)