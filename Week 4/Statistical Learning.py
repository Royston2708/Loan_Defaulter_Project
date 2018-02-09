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

# REGRESSION
np_age_1000 = df["age"].iloc[:1000].values
np_balance_1000 = df["balance"].iloc[:1000].values

plt.plot(np_age_1000, np_balance_1000 , marker = ".", linestyle = "none")
slope, intercept = np.polyfit(np_age_1000,np_balance_1000, 1)

print("slope is ", slope)
print("intercept is ", intercept)

x = np.array([0,100])
line = slope* x + intercept
plt.plot(x, line)
plt.show()