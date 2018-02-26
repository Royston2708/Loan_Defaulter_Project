from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn import svm
import xlsxwriter
#Reading the file into the system
file1 = pd.read_csv("/home/user/Downloads/portugese bank/bank-full-encoded.csv", sep=";" ,parse_dates= True)
print(file1.shape)

#Splitting into x,y and train and test data
y = file1["y"].values
x = file1.drop("y", axis = 1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 25)

#Running the SVM Classifier
svm_func = svm.SVC()
svm_func.fit(x_train, y_train)
svm_prediction = svm_func.predict(x_test)

print (confusion_matrix(y_test, svm_prediction))
print(classification_report(y_test, svm_prediction))

# Writing Output to Excel

writer = pd.ExcelWriter(path = "/home/user/Downloads/portugese bank/SVM and XGboost.xlsx", engine = 'xlsxwriter')
workbook = writer.book

svm_output = []
for i in svm_prediction:
    svm_output.append(i)

df_svm = pd.DataFrame(svm_output)
df_svm.to_excel(writer, sheet_name="SVM and XGboost", startcol=0, startrow=0)

#Running XGboost Algorithm
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train,y_train)
xgb_prediction = xgb.predict(x_test)

xgb_output = []
for i in xgb_prediction:
    xgb_output.append(i)

df_xgb = pd.DataFrame(xgb_output)
df_xgb.to_excel(writer, sheet_name="SVM and XGboost", startcol=3 , startrow=0)
print (confusion_matrix(y_test, xgb_prediction))
print(classification_report(y_test, xgb_prediction))


workbook.close()
writer.save()

print(len(svm_prediction), len(xgb_prediction))