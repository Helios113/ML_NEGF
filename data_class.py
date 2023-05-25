import numpy as np
from os.path import join
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, nargs="?", help="root")
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
        self.target = "/home/bailey/ML_NEGF/tar_dir"

        self.df = (
            pd.DataFrame()
        )  # [plane, vg, vd, location, type, iter, filedir, width, height, path]
        self.df_cond = (
            pd.DataFrame()
        )  # [plane, vg, vd, location, type, iter, filedir, width, height, path, mean, std, cmpPath, tarPath]

    def save(self, filename, condFlag=0):
        try:
            os.mkdir(join(self.target))
        except Exception as e:
            print("Found target dir")

        if condFlag == 0:
            self.df.to_csv(join(self.target, filename))
            print("Saved Dataframe to .csv!")
        else:
            self.df_cond.to_csv(join(self.target, filename))

    def read(self):
        # getting all files from all subdirectories and paths to said files
        print("Reading Files...")
        imgs = []
        name_as_lst = []
        VGnums = set()
        VDnums = set()
        target_prefix = "NEGF"

        filenum = 0
        filenames = []
        for root, dirs, files in os.walk(self.root):
            for name in files:
                if name.startswith(target_prefix):
                    path = os.path.join(root, name)
                    name = str(name).split(".txt")[0]  # Removing .txt at end of strings
                    name_as_lst = name.split("_")  # Spliting to
                    XY = name_as_lst.pop(0)[-2:]
                    name_as_lst.insert(0, XY)
                    Criteria = str(root).split("/")[-3]
                    Shape = str(root).split("/")[-2].split("x")
                    Height = Shape[0]
                    Width = Shape[1]
                    imgs.append(name_as_lst + [Criteria, Height, Width] + [path])

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
                "Height",
                "Width",
                "Path",
            ],
        )
        self.df = df
        print(self.df)

    def condition(self):
        # Condition should go through the entire self.array and condition based on the iteration number and data type
        # Get all entries with iterator == 1, each entire to find max and mid values of same result
        dataList = []
        unique_devices = self.df.query("Iteration == '1'")
        try:
            os.mkdir(join(self.target))
        except Exception as e:
            print("Found target dir")

        for index, row in unique_devices.iterrows():
            norm = [[0, 0], [0, 0]]
            # First find the max and mid value iteration for the given device file
            max = int(find_max_tier(self.df, row))
            mid = max // 2
            midResult = pd.DataFrame()

            # Now find the row that relates to that value and the other matching params
            i = 1
            print(row)
            while midResult.empty == True and mid >= 1:
                midResult = self.df.query(
                    "Plane == '{}' and VG == '{}' and VD == '{}' and Location == '{}' and Type == '{}' and Iteration == '{}' and Criteria == '{}' and Height == '{}' and Width == '{}'".format(
                        row["Plane"],
                        row["VG"],
                        row["VD"],
                        row["Location"],
                        row["Type"],
                        str(mid),
                        row["Criteria"],
                        row["Height"],
                        row["Width"],
                    )
                )
                mid = max - i
                i += 1

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

            if row["Type"] == "charge":
                data = np.log10(np.loadtxt(row["Path"]))
                chrg = 1
            else:
                data = np.loadtxt(row["Path"])
                chrg = 0

            norm[chrg][0] = data.mean()
            norm[chrg][1] = data.std()
            np.savetxt(
                join(self.target, row["Path"].split("/")[-1]),
                (data - norm[chrg][0]) / (norm[chrg][1]),
            )

            # Add to conditioned to list for dataframe later
            tmpList = row.values.tolist()
            tmpList.append(norm[chrg][0])
            tmpList.append(norm[chrg][1])

            # Get COMPARE data and normalise
            if midResult.iloc[0]["Type"] == "charge":
                data = np.log10(np.loadtxt(midResult.iloc[0]["Path"]))
                chrg = 1
            else:
                data = np.loadtxt(midResult.iloc[0]["Path"])
                chrg = 0
            np.savetxt(
                join(self.target, midResult.iloc[0]["Path"].split("/")[-1]),
                (data - norm[chrg][0]) / (norm[chrg][1]),
            )

            # Get TARGET data and normalise
            if maxResult.iloc[0]["Type"] == "charge":
                data = np.log10(np.loadtxt(maxResult.iloc[0]["Path"]))
                chrg = 1
            else:
                data = np.loadtxt(maxResult.iloc[0]["Path"])
                chrg = 0
            np.savetxt(
                join(self.target, maxResult.iloc[0]["Path"].split("/")[-1]),
                (data - norm[chrg][0]) / (norm[chrg][1]),
            )

            # Add paths to list
            tmpList.append(join(self.target, midResult.iloc[0]["Path"].split("/")[-1]))
            tmpList.append(join(self.target, maxResult.iloc[0]["Path"].split("/")[-1]))

            dataList.append(tmpList)

        self.df_cond = pd.DataFrame(
            dataList,
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
                "Path",
                "Mean",
                "STD",
                "cmpPath",
                "tarPath",
            ],
        )


a1 = Aggregator(args)
a1.read()
a1.save("dataframe.csv")
a1.condition()
a1.save("dataframe_conditioned.csv", 1)
