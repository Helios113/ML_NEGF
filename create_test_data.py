from os.path import join
import numpy as np

targetPath = '/home/bailey/ML_NEGF/testing_data'

critPath = ['abs_pot', 'avg_cur']
shapePath = ['3x9','3x12']

name = "NEGFXY"
vg = ['0.05','0.8']
vd = ['0.05','0.8']
type = ["charge", "pot"]
iteration = ['1','2','3','4','5']

for eachVG in vg:
    for eachVD in vd:
        for eachType in type:
            for eachIter in iteration:
                string = name + '_' + eachVG + '_' + eachVD + '_' + str(1) + '_' + eachType + '_' + eachIter + '.txt'
                mat = np.random.rand(3,3)
                np.savetxt(join(targetPath, 'abs_pot','3x12','ml_test',string),mat)
                np.savetxt(join(targetPath, 'abs_pot','3x9','ml_test',string),mat)
                np.savetxt(join(targetPath, 'avg_cur','3x12','ml_test',string),mat)
                np.savetxt(join(targetPath, 'avg_cur','3x9','ml_test',string),mat)