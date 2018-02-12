import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# quantiles

df = pd.read_csv("/home/user/Downloads/Data Sources/bank-full.csv", sep= ";", parse_dates=True)
print(df["age"].quantile([0.20,0.80]))
print(df.head(20))

# Example code of subplots

fig, axes = plt.subplots(nrows=3, ncols=1)
manage = df.loc[df["job"]=="management"]
df_manage= manage.iloc[:10,:]

df_manage.plot(ax=axes[0],kind="box")
plt.show()

#Creating an ECDF
#
# print(df["age"].values)
np_age = df["age"]
np_age_sort = np.sort(df["age"].values)
y= np.arange(1, len(np_age)+1)/len(np_age)

plt.plot(np_age, y ,linestyle="none", marker = ".")
plt.margins(0.02)
plt.show()

# REGRESSION (3rd argument in polyfit is the degree of the polynomial which is to be fitted to the 2 sets of data)
np_age_1000 = df["age"].iloc[2200:2300].values
np_balance_1000 = df["balance"].iloc[2200:2300].values

plt.plot(np_age_1000, np_balance_1000 , marker = ".", linestyle = "none")
slope, intercept = np.polyfit(np_age_1000,np_balance_1000, 1)

print("slope is ", slope)
print("intercept is ", intercept)

x = np.array([0,80])
line = slope* x + intercept
plt.plot(x, line)
plt.show()


# Bootstrapping using a function and without one

# With a function
def bs_values (data, func, size):
    bs_sample_generated = np.random.choice(data, len(data))
    bs_result = func(bs_sample_generated)
    return(bs_result)

# Without a function
bs_sample_age = np.random.choice(np_age_1000, size = len(np_age_1000))


# Generating multiple Bootstrap Replicates and stoing them

bs_replicates = np.empty(1000)
for i in range (1000):
    bs_replicates[i]= bs_values(np_age_1000, np.mean, size = len(np_age_1000))

plt.hist(bs_replicates, bins =50, normed = True)
plt.show()