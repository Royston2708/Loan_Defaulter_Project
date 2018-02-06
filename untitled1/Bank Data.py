import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from collections import Counter


sns.set()
df_data = pd.read_csv("/home/user/Downloads/Data Sources/bank-full.csv", sep= ";")

# df_data.iloc [::3, 0] = 10 will assign 10 to ever 3rd row starting from zero in the zero index column aka the first column
# df.head(5)/df.head(10) returns the first 5 rows and the first 10 rows respectively
print(df_data.head(10))

bins = df_data["job"].iloc[2000:4000].unique()

job = df_data["job"]
#Aliter to this is
# bins = df_data.iloc[2000:4000, 1]

print("Hello")
array = [item for item in df_data["job"]]
print(array)
# Aliter to above code is array = job.values
letter_counts = Counter(array)
hist_df = pd.DataFrame.from_dict(letter_counts, orient = 'index')
hist_df.plot(kind ='bar', figsize = (5, 5), fontsize = 7)
plt.show()


married_and_house = df_data[(df_data["marital"] == "married") & (df_data["housing"] == "yes")]
print(married_and_house)

age = [int(item) for item in married_and_house["age"]]
print(age)

plt.hist(age)
plt.show()

sns.swarmplot(x ="age", y="balance", data = df_data.iloc[:4000,:])
#df_data.iloc[:4000, :] returns a sub-dataframe with the first 4000 rows and all the columns
plt.show()

# Well done
