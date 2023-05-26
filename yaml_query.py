import pandas as pd
import yaml 

df = pd.read_csv('/home/kyle/ML_NEGF/test_csv_output/dataframe_conditioned.csv')
# skip header line
# drop columns that only have NaNs

with open("/home/kyle/ML_NEGF/query.yaml", "r") as stream:
    try:
        queries = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


#takes in a nested dictionary that is iterated through, the values in the nested dictionary tell us the type of query we are making and 
# the conditions in the query i.e. Height width if we are searching for a percentage and what that percentage is and if we are searching
# for VD of VG and the given value for VG/VD

training = queries["training"]
for key in queries:
    prevkey = key
    totalDF = pd.DataFrame()
    for key in queries[key]:
        args = queries[prevkey][key]
        if args["VG"] == "ALL":
            VG = df["VG"].unique()
        else:
            VG = str(args["VG"]).split(",")
            VG = [float(x) for x in VG]
        if args["VD"] == "ALL":
            VD = df["VD"].unique()
        else:
            VD = str(args["VD"]).split(",")
            VD = [float(x) for x in VD]
        ShapedDF = df[(df["Height"] == args["Height"]) & (df["Width"] == args["Width"]) & (df["Criteria"] == args["Criteria"])
                    & (df["VG"].isin(VG)) & (df["VD"].isin(VD))]
        condition = training[key]["Condition"]
        condition = str(condition)
        if condition.upper() == "ALL":
            selectedDf = ShapedDF
        elif "%" in condition:
            percentage = int(condition.split("%")[0])/100
            selectedDf = ShapedDF.sample(frac = percentage)
        elif "-" in condition:
            ranges = condition.split("-")
            start,stop = int(ranges[0]),int(ranges[1])
            selectedDf = ShapedDF[start:stop]
        elif "," in condition:
            strIndices = condition.split(",")
            intIndices = [int(i) for i in strIndices]
            selectedDf = ShapedDF.iloc[intIndices]
        elif condition.isnumeric():
            selectedDf = ShapedDF.iloc[int(condition)]
        elif condition.upper() == "FIRST AND LAST":
            selectedDf = pd.concat(ShapedDF.iloc[-1],ShapedDF.iloc[0])
        else:
            print("Invalid Input, please check user guide at www.example.com")
        index = selectedDf.index
    
        if condition.isnumeric():
            df = df.drop()
            pass
        else:
            df = df.drop(index)
        totalDF = pd.concat([totalDF,selectedDf])
        if prevkey == "training":
            trainingDF = totalDF
        else:
            testingDF = totalDF

print(trainingDF)
print(testingDF)