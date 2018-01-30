import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dict = {
    "country": [ "brazil", "russia", "india", "china", "south-africa"],
    "capital": ["brasilia", "moscow", "delhi", "beijing", "pretoria"],
    "area": [8.516, 17.10, 3.286, 9.597, 1.221],
    "population": [200.4, 143.5, 1252, 1357, 52.98]
}

brics = pd.DataFrame(dict)
brics.index = ["BR", "RU", "IN", "CH", "SA"]
print(brics,"\n\n")

census = pd.read_csv(filepath_or_buffer = "/home/user/Downloads/census.csv")
print(census["Age 80 to 84"])