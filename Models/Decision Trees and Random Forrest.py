import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xlsxwriter

#Reading the file into the system
file1 = pd.read_csv("/home/user/Downloads/portugese bank/bank-full-encoded.csv", sep=";" ,parse_dates= True)
print(file1.shape)

#Splitting into x,y and train and test data
y = file1["y"].values
x = file1.drop("y", axis = 1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 25)

#Running the Random Forrest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train, y_train)
rf_prediction = rf_classifier.predict(x_test)

print("\nThe Confusion Matrix is as follows:\n", confusion_matrix(y_test,rf_prediction))

print("\nThe Classification Report for the random forrest classifier is as follows:\n", classification_report(y_test, rf_prediction))

# Writing output to Excel

writer = pd.ExcelWriter(path = "/home/user/Downloads/portugese bank/Random Forrest.xlsx", engine = 'xlsxwriter')
workbook = writer.book

rf_output = []
for i in rf_prediction:
    rf_output.append(i)

df_rfoutput = pd.DataFrame(rf_output)
df_rfoutput.to_excel(writer, sheet_name="RandomForrest", startrow=0, startcol=0)

workbook.close()
writer.save()

print(len(rf_prediction))