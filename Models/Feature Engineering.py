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

def FeatureGeneration(column1, column2):

    column1_0 = (file1[str(column1)] == 0).values
    column2_0 = (file1[str(column2)] == 0).values
    GeneratedFeatureList = []
    for i, j in np.ndenumerate(column1_0):
        if column1_0[i] == True and column2_0[i] == True:
            GeneratedFeatureList.append("A0")
        elif column1_0[i] == True and column2_0[i] == False:
            GeneratedFeatureList.append("B0")
        elif column1_0[i] == False and column2_0[i] == False:
            GeneratedFeatureList.append("C0")
        else:
            GeneratedFeatureList.append("D0")

    np_GeneratedFeatureList = np.array(GeneratedFeatureList)
    UniqueVals = np.unique(np_GeneratedFeatureList)

    df_encoded_GenFeature = pd.get_dummies(data=np_GeneratedFeatureList, prefix=UniqueVals)

    return df_encoded_GenFeature

df_feature1 = FeatureGeneration(column1= "housing", column2= "duration_low")

new_df = pd.concat([x,df_feature1], axis= 1)

def TestResults(data,target):
    y = data[str(target)].values
    x = data.drop([str(target)], axis=1).values

    # Splitting into x,y and train and test data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)

    # Running the Random Forrest Classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(x_train, y_train)
    rf_prediction = rf_classifier.predict(x_test)

    return confusion_matrix(y_test, rf_prediction), classification_report(y_test, rf_prediction)

F1matrix , F1report = TestResults(data= new_df, target = "y")
print("The Confusion Matrix with 1 feature is\n", F1matrix)
print("\nThe Classification Report with 1 Feature is:\n", F1report)


# Generating housing and duration med
df_feature2 = FeatureGeneration(column1= "housing", column2= "duration_med")
new_df2 = pd.concat([new_df,df_feature2], axis= 1)

F2matrix , F2report = TestResults(data= new_df2, target = "y")
print("\nThe Confusion Matrix with 2 features generated is\n", F2matrix)
print("\nThe Classification Report with 2 Features generated is:\n", F2report)

#Generating housing and duration high
df_feature3 = FeatureGeneration(column1= "housing", column2= "duration_high")
new_df3 = pd.concat([new_df2,df_feature3], axis= 1)

F3matrix , F3report = TestResults(data= new_df3, target = "y")
print("\nThe Confusion Matrix with 3 features generated is\n", F3matrix)
print("\nThe Classification Report with 3 Features generated is:\n", F3report)

#Combining Tertiary Education and bluecollar job
df_feature4 = FeatureGeneration(column1= "edu_tertiary", column2= "job_bluecollar")
new_df4 = pd.concat([new_df3,df_feature4], axis= 1)

F4matrix , F4report = TestResults(data= new_df4, target = "y")
print("\nThe Confusion Matrix with 4 features generated is\n", F4matrix)
print("\nThe Classification Report with 4 Features generated is:\n", F4report)

#Combining marital and housing
df_feature5 = FeatureGeneration(column1= "marital", column2= "housing")
new_df5 = pd.concat([new_df4,df_feature5], axis= 1)

F5matrix , F5report = TestResults(data= new_df5, target = "y")
print("\nThe Confusion Matrix with 5 features generated is\n", F5matrix)
print("\nThe Classification Report with 5 Features generated is:\n", F5report)

