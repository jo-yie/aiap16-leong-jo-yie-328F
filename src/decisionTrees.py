# Decision Trees
# Adapted from https://www.datacamp.com/tutorial/decision-tree-classification-python

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Converting .csv into pandas df
df = pd.read_csv('data/cleaned_dataset.csv')

# Split dataset into features and target variable
X = df.drop(columns=['LungCancerOccurrence'])
y = df['LungCancerOccurrence']

# Split X and y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the model
clf = DecisionTreeClassifier()

# Fit model on the training data 
clf = clf.fit(X_train,y_train)

# Predict target variable using test features
y_pred = clf.predict(X_test)

# Print accuracy score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Optimising decision tree performance
print("Optimising by pre-pruning...")
clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=8)
clf2 = clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

# Print optimised decision tree accuracy score
print("Updated Accuracy:",metrics.accuracy_score(y_test, y_pred2))