# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# #
# dict = {
#     "country": [ "brazil", "russia", "india", "china", "south-africa"],
#     "capital": ["brasilia", "moscow", "delhi", "beijing", "pretoria"],
#     "area": [8.516, 17.10, 3.286, 9.597, 1.221],
#     "population": [200.4, 143.5, 1252, 1357, 52.98]
# }
#
# brics = pd.DataFrame(dict)
# brics.index = ["BR", "RU", "IN", "CH", "SA"]
# print(brics,"\n\n")
#
# census = pd.read_csv(filepath_or_buffer = "/home/user/Downloads/census.csv")
# print(census[["Age 25 to 29"  "Age "]])


# file = pd.read_csv("/home/user/Downloads/Data Sources/titanic.csv")
# print(file.head())
# print("\n", file[:10])

import pandas as pd
file = "/home/user/Downloads/Data Sources/battledeath.xlsx"
data = pd.ExcelFile(file)
print(data.sheet_names)

df1 = data.parse("2004")
print(df1.head)