import pandas as pd
import yaml
class query_data:
    def __init__(self, querypath,datapath):
        self.querypath = querypath
        self.datapath = datapath
        self.df = None
        self.queries = None


    
    def read_data(self):
        self.df = pd.read_csv(self.datapath)
        # skip header line
        # drop columns that only have NaNs
        queries = None
        with open(self.querypath, "r") as stream:
            try:
                self.queries = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


    def query_trans(self,args):
        # reads query from string and converts to consise info for df
        VG = None
        VD = None
        if str(args["VG"]).upper() == "ALL":
            VG = self.df["VG"].unique() # type: ignore
        elif "," in str(args["VG"]):
            VG = str(args["VG"]).split(",")
            VG = [float(x) for x in VG]
        elif "-" in str(args["VG"]):
            VG1 = str(args["VG"]).split("-")
            VG1 = [float(x) for x in VG1]
            VG = self.df["VG"].between(VG1[0], VG[1]) # type: ignore
        elif type(["VG"]) == float:
            VG = [args["VG"]]
            
        if str(args["VD"]).upper() == "ALL":
            VD = self.df["VD"].unique() # type: ignore
        elif "," in str(args["VD"]):
            VD = str(args["VD"]).split(",")
            VD = [float(x) for x in VD]
        elif "-" in str(args["VD"]):
            VD1 = str(args["VD"]).split("-")
            VD1 = [float(x) for x in VD1]
            VD = [x for x in self.df["VD"].unique() if (x >= VD1[0]) and (x <= VD1[1])]
            print(VD)
        elif type(args["VD"]) == float:
            VD = [args["VD"]]
        
        return VG, VD
   
    
    

    #takes in a nested dictionary that is iterated through, the values in the nested dictionary tell us the type of query we are making and 
    # the conditions in the query i.e. Height width if we are searching for a percentage and what that percentage is and if we are searching
    # for VD of VG and the given value for VG/VD
    def query_search(self):
        self.read_data()
        for key in self.queries:
            prevkey = key
            totalDF = pd.DataFrame()
            for key in self.queries[key]:
                args = self.queries[prevkey][key]
                
                VG, VD = self.query_trans(args)
                ShapedDF = self.df[(self.df["Height"] == args["Height"]) & (self.df["Width"] == args["Width"]) & (self.df["Criteria"] == args["Criteria"])
                            & (self.df["VG"].isin(VG)) & (self.df["VD"].isin(VD))]
                condition = args["Condition"]
                condition = str(condition)
                if condition.upper() == "ALL":
                    selectedDf = ShapedDF
                elif "%" in condition:
                    percentage = int(condition.split("%")[0])/100
                    selectedDf = ShapedDF.sample(frac = percentage)
                elif "-" in condition:
                    ranges = condition.split("-")
                    start,stop = int(ranges[0]),int(ranges[1])
                    selectedDf = ShapedDF[start:stop+1]
                elif "," in condition:
                    strIndices = condition.split(",")
                    intIndices = [int(i) for i in strIndices]
                    selectedDf = ShapedDF.iloc[intIndices]
                elif condition.isnumeric():
                    selectedDf = ShapedDF.iloc[int(condition)]
                elif condition.upper() == "FIRST AND LAST":
                    selectedDf = ShapedDF.iloc[[0,-1]]
                else:
                    print("Invalid Input, please check user guide at queryReadME.txt")
                index = selectedDf.index
            
                if condition.isnumeric():
                    self.df = self.df.drop(selectedDf.head()[0])
                    pass
                else:
                    self.df = self.df.drop(index)
                totalDF = totalDF.append(selectedDf)
                if prevkey == "training":
                    trainingDF = totalDF
                else:
                    testingDF = totalDF

        return trainingDF,testingDF
