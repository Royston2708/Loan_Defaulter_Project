
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests

url = "https://www.wikipedia.org/"
r = requests.get(url)
text = r.text

print (text)


df_election = pd.read_csv("/home/user/Downloads/Data Sources/US Election full data.csv")
print(df_election)

plt.hist(df_election["dem_share"], bins= 20)
plt.grid()
plt.xlabel("Instances")
plt.ylabel("Percentage")
plt.show()


