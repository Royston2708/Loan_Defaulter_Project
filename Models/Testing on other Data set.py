import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


file1 = pd.read_excel("/home/user/Downloads/Data Sources/Kaggle Datasets/Par_Data for Logistic Regression.xlsx", header = 1)
print(file1.shape)

y = file1["Default_On_Payment"]
x = file1.drop(["Customer_ID", "Default_On_Payment"], axis = 1)
x_columns = x.keys()
print(x_columns)

new_df = pd.DataFrame()

for key in x_columns:
    if x[str(key)].dtypes == 'object':
        uniques = x[str(key)].unique()
        uniques.sort()
        dumbo = pd.get_dummies(data = x[str(key)], prefix = uniques)
        new_df = pd.concat( [new_df, dumbo], axis = 1)


    elif x[str(key)].dtypes == None :
        x.drop([str(key)], axis = 1)

# To test for one hot encoding run this after the dumbo line : print(dumbo.head())
    else:
        new_df = pd.concat([new_df, x[str(key)]], axis = 1)

print(new_df.head())

#
#
# new_df = pd.concat(new_dict.values(), keys = new_dict.keys())
# print(new_df.head())
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 25)
# logreg = LogisticRegression()
# logreg.fit(x, y)

# prediction_logReg = logreg.predict(x_test)
# print("\nThe Confusion Matrix for the logreg Model is as follows: \n", confusion_matrix(y_test, prediction_logReg))
