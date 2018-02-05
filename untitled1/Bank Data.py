import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from collections import Counter

df_data = pd.read_csv("/home/user/Downloads/Data Sources/bank-full.csv", sep= ";", nrows = 1000)
print(df_data.head())
bins = df_data["job"].unique()

print("Hello")
array = [item for item in df_data["job"]]
letter_counts = Counter(array)
hist_df = pd.DataFrame.from_dict(letter_counts, orient = 'index')
hist_df.plot(kind ='bar', figsize = (7, 7), fontsize = 7)
plt.show()

# Well done
