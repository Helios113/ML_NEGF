import os
import pandas as pd
#getting all files from all subdirectories and paths to said files
imgs = []
name_as_lst = []
VGnums = set()
VDnums = set()
target_prefix = "NEGF"

for root, dirs, files in os.walk("/home/kyle/ML_NEGF/main_data_dir"):
    for name in files:
      if name.startswith(target_prefix):
        name = str(name).split(".txt")[0] #Removing .txt at end of strings
        name_as_lst = name.split("_") #Spliting to 
        XY = name_as_lst.pop(0)[-2:]
        name_as_lst.insert(0,XY)
        Critiria = str(root).split("/")[-3]
        Shape = str(root).split("/")[-2].split("x")
        Height = Shape[0]
        Width = Shape[1]
        imgs.append(name_as_lst + [Critiria,Height,Width] +[os.path.join(root, name)])




imgsdf = pd.DataFrame(imgs,columns=['Plane','VG','VD','Location',"Type","Iteration","Critiria","Height","Width","Path"])
imgsdf.to_csv("/home/kyle/ML_NEGF/test_csv_output/test.csv")

print("The data frame starts with {} rows \n".format(len(imgsdf.index)))

def extended_query(string,df):
   args = string.split(" ")
   #The first two args in the string are height and width, these are the main thing we search by
   height,width = args[0],args[1]
   if len(args) == 3 and (args[2].upper() == "ALL"):
      #Handels special case of ALL
      selectedDf = df.query("Height == '{}' and Width == '{}'".format(height,width))
   else:
      #The input string is formatted to tell us what the selected querry is, what the search parameters are 
      # if we are using VG or VD and what the is the given value for VG/VD
      selection,VGVD = args[2],args[3] 
      selection = selection.split(":") #The first part of selection tells us if it is a percentage,range or specific indices and the second part is the value of said selection i.e. 50%
      selection[0] = selection[0].upper()
      VGVD = VGVD.split(":") #First element tells us the if it is VG or VD second element tells us the value
      shapeDF = df.query("Height == '{}' and Width == '{}' and {} == '{}'".format(height,width,VGVD[0],VGVD[1]))
      if selection[0].startswith("P"): #Percentage
         percent = int(selection[1])/100 
         selectedDf = shapeDF.sample(frac = percent)
      elif selection[0].startswith("R"): #Range
         ranges = selection[1].split("to")
         start,stop = int(ranges[0]),int(ranges[1])
         selectedDf = shapeDF[start:stop]
      elif selection[0].startswith("S"): #Specific values
         strIndices = selection[1].split(",")
         intIndices = [int(i) for i in strIndices]
         selectedDf = shapeDF.iloc[intIndices]
      else:
         print("Error: Invalid data selection must choose from, Percentage, Range, Specific.")

   index = selectedDf.index #Finds the indices of selected data so that it can be removed from df
   df = df.drop(index) #Removes queried data from dataframe
   print(df)
   print("\n")
   return selectedDf

querieddf = extended_query("3 9 Percentage:50 VD:0.05",imgsdf)
print("The queried dataframe contains {} rows".format(len(querieddf.index)))
print(querieddf)

