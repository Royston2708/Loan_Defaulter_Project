import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# The Objective of this code is to build a K-Nearest Neighbours Classifier for the Bank Data that we have
file1 = pd.read_csv("/home/user/Downloads/portugese bank/bank-full-encoded.csv", sep=";" ,parse_dates= True)
print(file1.shape)

# Setting Plot Style To Seaborn
sns.set()

#Creating a Countplot of the count of people who are middle aged vs whether or not they were approved for a loan
sns.countplot("age_mid", hue = "y", data = file1, palette = "RdBu")
plt.xticks([0,1], ["NOT MIDDLE AGED","MIDDLE AGED"])
plt.title("AGE")
plt.show()

# Determining the correlation coefficient
print(file1["age_mid"].values)
print("\nThe Correlation Coefficient between middle age and approval for a loan is = ")
print(np.corrcoef(file1["age_mid"].values, file1["y"].values))

# Building a KNN Predictor model for the data
y = file1["y"].values
x = file1.drop("y", axis = 1).values

knn = KNeighborsClassifier(n_neighbors= 10)
knn.fit(x,y)

np.random.seed(50)
x_new = [np.random.randint(0,2,26)]
print(x_new)

prediction = knn.predict(x_new)
print("The label of the new value is = ",prediction)