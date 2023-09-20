# Bangma Disease Model
### Author: Hamza Mohammed

<p align="center">
  <img src="lab_test_tube.jpg" alt="Image" width="50%">
</p>

## Synopsis
I developed this disease prediction model to accurately predict various diseases based on medical records. This project is important as disease prediction algorithms increasingly play a vital role in healthcare, aiding in early diagnosis, timely interventions, and improved patient outcomes. This documentation provides a clear overview of the project's workflow, from data preparation to model training and evaluation.



## Table of Contents

* Data Preparation
* Training the Model
* Decision Tree Classifier
* Random Forest Classifier
* Evaluating the Model
* Conclusion

## Imports
The first step in this project is to import the necessary libraries that will be used for data manipulation, visualization, and machine learning. The following libraries are imported:


import numpy as np<br>
import pandas as pd<br>
import seaborn as sb<br>
import matplotlib.pyplot as plt<br>
from warnings import filterwarnings<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.tree import DecisionTreeClassifier<br>
from sklearn.ensemble import RandomForestClassifier<br>
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix<br>


## Data Preparation
Once the necessary libraries are imported, the next step is to prepare the data for analysis and modeling. Data preparation involves the following steps:




### Load the training and testing data
train = pd.read_csv("Train_data.csv")
test = pd.read_csv('Test_data.csv')



### Display the first 10 rows of the training data
train.iloc[:10]




### Check for missing values in the training data
A.isna().sum()

### Splitting Data
Split the data into training and testing sets. The training set is used to train the machine learning model, and the testing set is used to evaluate its performance.

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

## Training the Model
In this project, two different machine learning models are trained: a Decision Tree Classifier and a Random Forest Classifier.

### Decision Tree Classifier
A Decision Tree Classifier is a machine learning algorithm that uses a tree-like structure to classify data. It splits the data recursively based on feature values.



#### Create a Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)

#### Train the Decision Tree model
model_dtc = dtc.fit(xtrain, ytrain)

3.2 Random Forest Classifier
A Random Forest Classifier is an ensemble learning algorithm that uses multiple decision trees to make predictions.


### Create a Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)

#### Train the Random Forest model
model_rfc = rfc.fit(xtrain, ytrain)

## Evaluating the Model
After training the models, it's essential to evaluate their performance on the testing set using various metrics such as accuracy, F1 score, classification report, and confusion matrix.



#### Evaluate the Decision Tree model
tr_pred_dtc = model_dtc.predict(xtrain)
ts_pred_dtc = model_dtc.predict(xtest)

print("The train accuracy is:", accuracy_score(ytrain, tr_pred_dtc))
print("The test accuracy is:", accuracy_score(ytest, ts_pred_dtc))

print("Classification Report:")
print(classification_report(ytest, ts_pred_dtc))

print("Confusion Matrix:")
print(confusion_matrix(ytest, ts_pred_dtc))

#### Evaluate the Random Forest model
tr_pred_rfc = model_rfc.predict(xtrain)
ts_pred_rfc = model_rfc.predict(xtest)

print("Training accuracy is:", accuracy_score(ytrain, tr_pred_rfc))
print("Testing accuracy is:", accuracy_score(ytest, ts_pred_rfc))

## Conclusion
In this project, I used two machine learning models, the Decision Tree Classifier and the Random Forest Classifier for disease prognosis. Here are some key takeaways:

* The Decision Tree Classifier is simple and easy to interpret but can be prone to overfitting.
* The Random Forest Classifier is more complex, less prone to overfitting, but computationally expensive.
* The choice of the best algorithm depends on the specific dataset and desired performance.
* The models achieved high accuracy in this project, indicating their potential for disease diagnosis.

## How to Execute the Project

To run this project, follow these steps:

* Make sure you have Python installed on your system.

* Install the required libraries using pip:


pip install numpy pandas seaborn matplotlib scikit-learn

* Place your training data in a file named "Train_data.csv" and your testing data in a file named "Test_data.csv" in the project directory.

* Execute the provided code snippets in a Python environment to load, prepare, train, and evaluate the modelsusing:


python bangma.py









