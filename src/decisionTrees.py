# Decision Trees
# Adapted from https://www.datacamp.com/tutorial/decision-tree-classification-python

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Ingesting dataset into pandas dataframe and splitting into testing and training datasets
df = pd.read_csv('data/cleaned_dataset.csv')
x = df.drop(columns=['Lung Cancer Occurrence'])
y = df['Lung Cancer Occurrence']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Optimising Decision Tree Performance
print("Optimising by pre-pruning...")
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train, y_train)
y_pred = clf.preditct(X_test)

print("Updated Accuracy:",metrics.accuracy_score(y_test, y_pred))