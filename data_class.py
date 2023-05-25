import numpy as np
from os.path import join
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, nargs='?', default='/tar_dir/',
                    help='root')
args = parser.parse_args()

def find_max_tier(df, row):
    result = df.query(
        "Plane == '{}' and VG == '{}' and VD == '{}' and Location == '{}' and Type == '{}' and Criteria == '{}' and Height == '{}' and Width == '{}'".format(
            row["Plane"],
            row["VG"],
            row["VD"],
            row["Location"],
            row["Type"],
            row["Criteria"],
            row["Height"],
            row["Width"],
        )
    )

    return result.max()["Iteration"]


class Aggregator:
    def __init__(self, args):
        self.root = args.root
        self.target = args.target

        self.df = (
            pd.DataFrame()
        )  # [plane, vg, vd, location, type, iter, filedir, width, height, path]
        self.df_cond = (
            pd.DataFrame()
        )  # [plane, vg, vd, location, type, iter, filedir, width, height, mean, std, cmpPath, tarPath, path]

        self.avgmappings = {"avg_cur": 0, "abs_pot": 1}
        self.planemappings = {"xy": 0, "yz": 1, "zx": 2}
        self.typemappings = {"charge": 0, "pot": 1}

    def read(self):
        # getting all files from all subdirectories and paths to said files
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
                    name = str(name).split(".txt")[0]  # Removing .txt at end of strings
                    name_as_lst = name.split("_")
                    XY = name_as_lst.pop(0)[-2:]
                    name_as_lst.insert(0, XY)
                    imgs.append(name_as_lst + str(root).split("/")[-3:])

        df = pd.DataFrame(
            imgs,
            columns=[
                "Plane",
                "VG",
                "VD",
                "Location",
                "Type",
                "Iteration",
                "Criteria",
                "Shape",
                "Index",
            ],
        )
        self.df = df

    def condition(self):
        # Condition should go through the entire self.array and condition based on the iteration number and data type
        # Get all entries with iterator == 1, each entire to find max and mid values of same result

        norm = [[0, 0], [0, 0]]
        unique_devices = self.df.query("Iterator == '1'")

        for index, row in self.df.iterrows():
            # First find the max and mid value iteration for the given device file
            max = find_max_tier(self.df, row)
            mid = max // 2

            # Now find the row that relates to that value and the other matching params
            midResult = self.df.query(
                "Plane == '{}' and VG == '{}' and VD == '{}' and Location == '{}' and Type == '{}' and Iteration == '{}' and Criteria == '{}' and Height == '{}' and Width == '{}'".format(
                    row["Plane"],
                    row["VG"],
                    row["VD"],
                    row["Location"],
                    row["Type"],
                    mid,
                    row["Criteria"],
                    row["Height"],
                    row["Width"],
                )
            )
            maxResult = self.df.query(
                "Plane == '{}' and VG == '{}' and VD == '{}' and Location == '{}' and Type == '{}' and Iteration == '{}' and Criteria == '{}' and Height == '{}' and Width == '{}'".format(
                    row["Plane"],
                    row["VG"],
                    row["VD"],
                    row["Location"],
                    row["Type"],
                    max,
                    row["Criteria"],
                    row["Height"],
                    row["Width"],
                )
            )

            if row["Type"] == "Charge":
                data = np.log10(np.loadtxt(row["Index"]))
                chrg = 1
            else:
                data = np.loadtxt(row["Index"])
                chrg = 0

            norm[chrg][0] = data.mean()
            norm[chrg][1] = data.std()
            np.savetxt(
                join("/home/bailey/ML_NEGF/tar_dir", row["Path"].split("/")[-1]),
                (data - norm[chrg][0]) / (norm[chrg][1]),
            )

            # Add to conditioned to list for dataframe later
            list = row.values.tolist().append(norm[chrg][0] + norm[chrg][1])

            # Reset norm lists
            norm = [[0, 0], [0, 0]]

            # Get COMPARE data and normalise
            if midResult["Type"] == "Charge":
                data = np.log10(np.loadtxt(midResult["Index"]))
                chrg = 1
            else:
                data = np.loadtxt(midResult["Index"])
                chrg = 0
            np.savetxt(
                join("/home/bailey/ML_NEGF/tar_dir", midResult["Path"].split("/")[-1]),
                (data - norm[chrg][0]) / (norm[chrg][1]),
            )

            # Reset norm lists
            norm = [[0, 0], [0, 0]]

            # Get TARGET data and normalise
            if maxResult["Type"] == "Charge":
                data = np.log10(np.loadtxt(maxResult["Index"]))
                chrg = 1
            else:
                data = np.loadtxt(row["Index"])
                chrg = 0
            np.savetxt(
                join("/home/bailey/ML_NEGF/tar_dir", maxResult["Path"].split("/")[-1]),
                (data - norm[chrg][0]) / (norm[chrg][1]),
            )

            # Add paths to list
            list.append(
                join("/home/bailey/ML_NEGF/tar_dir", midResult["Path"].split("/")[-1])
                + join("/home/bailey/ML_NEGF/tar_dir", maxResult["Path"].split("/")[-1])
            )

        self.df_cond = pd.DataFrame(
            list,
            columns=[
                "Plane",
                "VG",
                "VD",
                "Location",
                "Type",
                "Iteration",
                "Criteria",
                "Height",
                "Width",
                "Mean",
                "STD",
                "cmpPath",
                "tarPath" "Path",
            ],
        )