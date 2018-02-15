import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# Permutation Sample Function
def permutation_sample(data1, data2):
    # store data together
    data = np.concatenate((data1,data2))
    #Permute the concatenated data
    permuted_data = np.random.permutation(data)

    perm_sample_1 = permuted_data[len(data1)]
    perm_sample_2 = permuted_data[len(data2)]

    return perm_sample_1, perm_sample_2


file1 = pd.read_csv("/home/user/Downloads/portugese bank/bank-full-encoded.csv", sep=";" ,parse_dates= True)
good_edu = file1
print(good_edu.values, "\n\n")

#Values serves as the aggregator, columns is the seperator.

pivot = pd.pivot_table(good_edu, values = "balance_med", index= ["edu_secondary","age_mid"], columns= "y", aggfunc= np.sum)
print("The pivot Table which classifies individuals on the basis of medium balance, secondary education and medium age",
      "\n", pivot, "\n")

dim = np.array(pd.DataFrame(file1).shape)
print("The Dataframe has the following dimentsions\nrows = ", str(dim[0]), "; columns = ", str(dim[1]))


