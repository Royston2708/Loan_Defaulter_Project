import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from collections import Counter


sns.set()
df_data = pd.read_csv("/home/user/Downloads/Data Sources/bank-full.csv", sep= ";")
print(df_data.head())
bins = df_data["job"].unique()

print("Hello")
array = [item for item in df_data["job"]]
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


# Well done
