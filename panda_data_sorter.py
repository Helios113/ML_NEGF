import os
import pandas as pd

# getting all files from all subdirectories and paths to said files
imgs = []
name_as_lst = []
VGnums = set()
VDnums = set()
target_prefix = "NEGF"

for root, dirs, files in os.walk("/home/bailey/ML_NEGF/main_data_dir"):
    for name in files:
        if name.startswith(target_prefix):
            name = str(name).split(".txt")[0]  # Removing .txt at end of strings
            name_as_lst = name.split("_")
            XY = name_as_lst.pop(0)[-2:]
            name_as_lst.insert(0, XY)
            Criteria = str(root).split("/")[-3]
            Shape = str(root).split("/")[-2].split("x")
            Height = Shape[0]
            Width = Shape[1]
            imgs.append(
                name_as_lst + [Criteria, Height, Width] + [os.path.join(root, name)]
            )


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

iterationQuery = df.query("Iteration == '1'")
# print(iterationQuery)

for index, row in iterationQuery.iterrows():
    uniqueQuery = df.query(
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
    print(uniqueQuery.max()["Iteration"])
