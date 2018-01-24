import numpy as np
import matplotlib as mtlib

height = [1.72, 1.69, 1.85, 2.01, 1.79]
np_height = np.array(height)
print("heights are", np_height)
weight= [65.4, 59.2, 63.6, 88.4, 68.7]
np_weight= np.array(weight)
print("weights are", np_weight, "\n")
bmi = np_weight/np_height**2
print("bmi data is", bmi)
print(bmi[bmi > 22])

np_2d = np.array([height, weight])
print(np_2d,"\n")

element_1 = np_2d[0][3]
print("first element selected is", element_1, "\n")

group_1 = np_2d[:, 1:3]
print("group one has john and sean with the following data\n", group_1)