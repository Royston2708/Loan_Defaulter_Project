from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn import svm

file1 = pd.read_csv("/home/user/Downloads/portugese bank/bank-full-encoded.csv", sep=";" ,parse_dates= True)
print(file1.shape)

y = file1["y"].values
x = file1.drop("y", axis = 1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 25)

svm_func = svm.SVC()
svm_func.fit(x_train, y_train)
svm_prediction = svm_func.predict(x_test)

print (confusion_matrix(y_test, svm_prediction))
print(classification_report(y_test, svm_prediction))