import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


file1 = pd.read_excel("/home/user/Downloads/Data Sources/Kaggle Datasets/Par_Data for Logistic Regression.xlsx", header = 1)
print(file1.shape)

y = file1["Default_On_Payment"]
x = file1.drop(["Customer_ID"], axis = 1)
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
print(new_df.keys())
# Splitting Data into Test and Train
y_new = new_df["Default_On_Payment"].values
x_new = new_df.drop(["Default_On_Payment","Count"], axis= 1).values
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.3, random_state= 25)

# Running LogReg
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

logReg_prediction = logreg.predict(x_test)
print("\nThe Confusion Matrix for the logreg Model is as follows: \n", confusion_matrix(y_test, logReg_prediction))

# Running Random Forrest
rf = RandomForestClassifier()

rf.fit(x_train,y_train)
rf_prediction = rf.predict(x_test)
print("\nThe Confusion Matrix for the Random Forrest Algorithm is as follows:\n",confusion_matrix(y_test, rf_prediction))
print("\nThe Classification report for the Random Forrest Model is:\n",classification_report(y_test,rf_prediction))

# Running SVM

svm = svm.SVC()
svm.fit(x_train, y_train)
svm_prediction = svm.predict(x_test)
print("\nThe Confusion Matrix for SVM is as follows:\n",confusion_matrix(y_test, svm_prediction))
print("\nThe Classification report for the SVM Model is:\n",classification_report(y_test,svm_prediction))