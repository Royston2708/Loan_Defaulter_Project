import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    global prediction, test_data ,train_Data, file1
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

    # Splitting Data into Training set and test Set
    train_Data = file1.iloc[:32000,:]
    test_data =file1.iloc[32000:,:]

    # Building a KNN Predictor model for the data
    y = train_Data["y"].values
    x = train_Data.drop("y", axis = 1).values

    knn = KNeighborsClassifier(n_neighbors= 10)
    knn.fit(x,y)

    #Predicting using our aldready labelled data set

    prediction = knn.predict(test_data.drop("y", axis = 1))
    count_0 = 0
    count_1 = 0
    for i in prediction:
        if i == 0 :
            count_0 += 1
        else:
            count_1 += 1
        print(i)
    return count_0, count_1

zero , one = main()
print("count of zeroes is =", zero,"\n count of ones is =", one)

# Accuracy of the model

wrong_prediction = 0
correct_prediction = 0
i = 0
for element in prediction:
    if element != test_data["y"].iloc[i]:
        wrong_prediction +=1
    else:
        correct_prediction += 1
    i+= 1

print("Number of incorrect predictions =", wrong_prediction,"\nNumber of correct predictions =", correct_prediction)
print("Accuracy of model =", (correct_prediction/(wrong_prediction+correct_prediction))*100, "%")

# Aliter to above code to predict model accuracy is to use train_test_split and knn.score

# Logistic Regression Model

y_new = file1["y"].values
x_new = file1.drop("y", axis = 1).values

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.3, random_state= 25)

logreg = LogisticRegression(C = 2)
logreg.fit(x_train, y_train)

y_prediction = logreg.predict(x_test)

# Count of zeroes and Ones
count_0_reg = 0
count_1_reg = 0
for i in y_prediction:
    if i == 0 :
        count_0_reg += 1
    else:
        count_1_reg += 1

print("\n\n", count_0_reg, count_1_reg)

# Accuracy of LogReg
wrong_prediction_reg = 0
correct_prediction_reg = 0
i = 0
for element in y_prediction:
    if element != y_new[i]:
        wrong_prediction_reg +=1
    else:
        correct_prediction_reg += 1
    i+= 1

print("Number of incorrect predictions =", wrong_prediction_reg,"\nNumber of correct predictions =", correct_prediction_reg)
print("Accuracy of model =", (correct_prediction_reg/(wrong_prediction_reg+correct_prediction_reg))*100, "%")
