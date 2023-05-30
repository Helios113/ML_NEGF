The query search takes in a .yaml file. This file is a yaml nested dictionary in the following form:
The first layer of the nested dictionary should have two keys in the following order training and testing.
These keys should contain the queries describing what is needed for the training data and the testing data.
It is possible to have mutiple queries inside the trainging and testing.

The format of the queries goes as follows:
There are 6 keys, Height, Width, Criteria, Condition, VG, VD
The values for Height and Width are both integers describing the height and width of the device.
The value for Criteria describes wether the value is abs_pot or avg_cur
The value for Condition is dependant on how you want to take data from the query, the possible choices are:
    Percentage: Which would be in the form of number% i.e. 50%
    Range: Which would be in the form start-stop i.e. 0-2
    Mutiple Indices: Which would be in the form index,index,index i.e. 1,2,3
    Specific Index: Which would be in the form of just the index i.e. 0
    ALL: In the form of ALL, this is the equivlant of 100% 
The values for VD and VG are as follows:
    Range: Which would be in the form start-stop i.e. 0.05-0.8
    Specific values: Which would be in the form value,value,value i.e. 0.05,0.425,0.8
    Specific value: Which would be in the form of just the value i.e. 0.05
    ALL: Which is in the form ALL and is ALL possible values of VD/VG

The query search works by first finding the dataset for training, removing all training data from the main dataset, then using queries
the leftover data in the main dataset to find the testing values.
Because of this it might be a good idea to use ALL in testing for conditions where you want all data that is left after training.

An example query is shown below:

training:
    query:
        Height: 3
        Width: 9
        Criteria: abs_pot
        Condition: 50%
        VD: 0.05-0.8
        VG: ALL
    query2:
        Height: 3
        Width: 12
        Criteria: abs_pot
        Condition: 0
        VD: 0.8
        VG: ALL


testing:
    query:
        Height: 3
        Width: 9
        Condition: ALL
        VD: ALL
        Criteria: abs_pot
        VG: ALL
    query2:
        Height: 3
        Width: 12
        Condition: ALL
        VD: 0.8
        Criteria: abs_pot
        VG: ALL