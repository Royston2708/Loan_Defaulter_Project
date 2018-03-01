import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xlsxwriter

#Reading the file into the system
file1 = pd.read_csv("/home/user/Downloads/portugese bank/bank-full-encoded.csv", sep=";" ,parse_dates= True)
print(file1.shape)
#
# feature1 = FeatureUnion([("housing", file1["housing"]),("low duration",file1["duration_low"])])
y= file1["y"]
x = file1

housing_0 = (file1["housing"]==0).values
duration_low_0 = (file1["duration_low"]==0).values
duration_med_0 = (file1["duration_med"]==0).values
duration_high_0 = (file1["duration_high"]==0).values


feature1 = []
for i,j in np.ndenumerate(housing_0):
    if housing_0[i] == True and duration_low_0[i]== True:
        feature1.append("A")
    elif housing_0[i] == True and duration_low_0[i] ==False:
        feature1.append("B")
    elif housing_0[i] ==False and duration_low_0[i] ==False:
        feature1.append("C")
    else:
        feature1.append("D")


np_feature1 = np.array(feature1)
uniques = np.unique(np_feature1)


df_encoded_feature = pd.get_dummies(data = np_feature1, prefix= uniques)
new_df = pd.concat([x,df_encoded_feature], axis= 1)
print(new_df.head())

y_new = new_df["y"].values
x_new = new_df.drop(["y"], axis = 1).values


#Splitting into x,y and train and test data

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.3, random_state= 25)

#Running the Random Forrest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train, y_train)
rf_prediction = rf_classifier.predict(x_test)

print("\nThe Confusion Matrix is as follows:\n", confusion_matrix(y_test,rf_prediction))

print("\nThe Classification Report for the random forrest classifier is as follows:\n", classification_report(y_test, rf_prediction))


#CREATING MARITAL FEATURE
marital_0 = (file1["marital"]==0).values
feature3 = []
for i,j in np.ndenumerate(marital_0):
    if marital_0[i] == True and duration_low_0[i]== True:
        feature3.append("Q")
    elif marital_0[i] == True and duration_low_0[i] ==False:
        feature3.append("R")
    elif marital_0[i] == False and duration_low_0[i] ==False:
        feature3.append("S")
    else:
        feature3.append("T")

np_feature3 = np.array(feature3)
uniques3 = np.unique(np_feature3)
df_encoded_feature3 = pd.get_dummies(data = np_feature3, prefix= uniques3)
new_df_3 = pd.concat([new_df,df_encoded_feature3], axis= 1)
print(new_df_3.head())

y_new3 = new_df_3["y"].values
x_new3 = new_df_3.drop("y", axis= 1).values


x_train3, x_test3, y_train3, y_test3 = train_test_split(x_new3, y_new3, test_size= 0.3, random_state= 25)

#Running the Random Forrest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train3, y_train3)
rf_prediction3 = rf_classifier.predict(x_test3)

print("\nThe Confusion Matrix is as follows:\n", confusion_matrix(y_test2,rf_prediction2))

print("\nThe Classification Report for the random forrest classifier is as follows:\n", classification_report(y_test2, rf_prediction2))
