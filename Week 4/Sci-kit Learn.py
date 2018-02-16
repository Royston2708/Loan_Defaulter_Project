import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets

plt.style.use("ggplot")

iris = datasets.load_iris()
print("The dataset has the following number of rows x columns", iris.data.shape)
print(iris.keys())

#Building Data frame and doing some EDA
x = iris.data
y= iris.target

df = pd.DataFrame(x, columns = iris.feature_names)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)
print(knn.fit(iris["Data"],iris["target"]))
