# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10.find the accuracy of our model and predict the require values.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: sudharsan.s
RegisterNumber:  24009664
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\sudharshan\\Downloads\\Employee.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

print(data["left"].value_counts())


le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


new_data = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
print(f"Prediction for new data: {dt.predict(new_data)}")


plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=x.columns, class_names=["Stayed", "Left"], filled=True)
plt.show()
*/
```

## Output ![Screenshot 2024-11-28 203824](https://github.com/user-attachments/assets/b7b02941-bc30-4292-89a4-3cfa2f0e19e9)
![Screenshot 2024-11-28 203947](https://github.com/user-attachments/assets/387b527f-8290-4be1-9611-2d9f23266b9a)
![Screenshot 2024-11-28 204000](https://github.com/user-attachments/assets/fa33c678-1265-44b6-b51a-e2eac6014404)
![Screenshot 2024-11-28 204018](https://github.com/user-attachments/assets/d7d148cc-49e9-484e-a8f1-56353de9a082)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
