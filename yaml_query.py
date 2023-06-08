import pandas as pd
import yaml


class query_data:
    def __init__(self, datapath):
        self.datapath = datapath
        self.df = None
        self.read_data()
        self.queries = {}

    def read_data(self):
        self.df = pd.read_csv(
            self.datapath,
            header=0,
            index_col=False,
            dtype={
                0: 'Int64',
                1: str,
                2: float,
                3: float,
                4: int,
                5: str,
                6: int,
                7: int,
                8: str,
                9: str,
                10:str,
                11: str,
                12: str,
                13: str,
                14: float,
                15: float,
                16: float,
                17: float,
            }
        )
        # skip header line
        # drop columns that only have NaNs

    def query_trans(self, args):
        # reads query from string and converts to consise info for df
        gate = None
        drain = None
        Location = None
        if str(args["VG"]).upper() == "ALL":
            gate = self.df["VG"].unique()  # type: ignore
        elif "," in str(args["VG"]):
            gate = str(args["VG"]).split(",")
            gate = [float(x) for x in gate]
        elif "-" in str(args["VG"]):
            VG1 = str(args["VG"]).split("-")
            VG1 = [float(x) for x in VG1]
            gate = self.df["VG"].between(VG1[0], VG[1])  # type: ignore
        elif type(args["VG"]) == float:
            gate = [args["VG"]]

        if str(args["VD"]).upper() == "ALL":
            drain = self.df["VD"].unique()  # type: ignore
        elif "," in str(args["VD"]):
            drain = str(args["VD"]).split(",")
            drain = [float(x) for x in drain]
        elif "-" in str(args["VD"]):
            VD1 = str(args["VD"]).split("-")
            VD1 = [float(x) for x in VD1]
            drain = [x for x in self.df["VD"].unique() if (x >= VD1[0]) and (x <= VD1[1])]
        elif type(args["VD"]) == float:
            drain = [args["VD"]]

        if str(args["Location"]).upper() == "ALL":
            Location = self.df["Location"].unique()
        elif "," in str(args["Location"]):
            Location = str(args["Location"]).split(",")
            Location = [float(x) for x in Location]
        elif "-" in str(args["Location"]):
            Loc1 = str(args["Location"]).split("-")
            Location = [x for x in range(Loc1[0], Loc1[1])]
        elif type(args["Location"]) == int:
            Location = [args["Location"]]
        return gate, drain, Location

    def query_search(self, querypath):
        self.querypath = querypath
        with open(self.querypath, "r") as stream:
            try:
                self.queries = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        trainingDF = list()
        testingDF = list()

        self.read_data()
        for key in self.queries:
            prevkey = key
            for key in self.queries[key]:
                totalDF = pd.DataFrame()
                selectedDf = pd.DataFrame()
                args = self.queries[prevkey][key]
                VG, VD, Location = self.query_trans(args)
                ShapedDF = self.df[
                    (self.df["Height"] == args["Height"])
                    & (self.df["Width"] == args["Width"])
                    & (self.df["Criteria"] == args["Criteria"])
                    & (self.df["VG"].isin(VG))
                    & (self.df["VD"].isin(VD))
                    & (self.df["Plane"] == args["Plane"])
                    & (self.df["Location"].isin(Location))
                ]
                condition = args["Condition"]
                condition = str(condition)
                if condition.upper() == "ALL":
                    selectedDf = ShapedDF
                elif "%" in condition:
                    percentage = int(condition.split("%")[0]) / 100
                    selectedDf = ShapedDF.sample(frac=percentage)
                elif "-" in condition:
                    ranges = condition.split("-")
                    start, stop = int(ranges[0]), int(ranges[1])
                    selectedDf = ShapedDF[start : stop + 1]
                elif "," in condition:
                    strIndices = condition.split(",")
                    intIndices = [int(i) for i in strIndices]
                    selectedDf = ShapedDF.iloc[intIndices]
                elif condition.isnumeric():
                    selectedDf = ShapedDF.iloc[int(condition)]
                elif condition.upper() == "FIRST AND LAST":
                    selectedDf = ShapedDF.iloc[[0, -1]]
                else:
                    print("Invalid Input, please check user guide at queryReadME.txt")
                index = selectedDf.index

                if condition.isnumeric():
                    self.df = self.df.drop(selectedDf.head()[0])
                    selectedDf = pd.DataFrame(selectedDf).transpose()
                    totalDF = pd.concat([totalDF, selectedDf])
                else:
                    self.df = self.df.drop(index)
                    totalDF = pd.concat([totalDF, selectedDf])
                if prevkey == "training":
                    trainingDF.append(totalDF)
                else:
                    testingDF.append(totalDF)

        return trainingDF, testingDF


# query = query_data("query.yaml","cond_data/dataframe_conditioned.csv")
# train,test = query.query_search()

# print(train)
# print(test)
