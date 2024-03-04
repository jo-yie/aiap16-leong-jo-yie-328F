# Logistic Regression 
# Adapted from https://www.datacamp.com/tutorial/understanding-logistic-regression-python

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import seaborn as sns

# Converting .csv into pandas df
df = pd.read_csv('data/cleaned_dataset.csv')

# Split dataset into features and target variable
X = df.drop(columns=['LungCancerOccurrence'])
y = df['LungCancerOccurrence']

# Split X and y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the model
logreg = LogisticRegression()

# Fit model on the training data 
logreg.fit(X_train, y_train)

# Predict target variable using test features
y_pred = logreg.predict(X_test)

# Model evaluation with confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# Model evaluation with classification_report 
target_names = ['No Lung Cancer', 'Lung Cancer']
print(classification_report(y_test, y_pred, target_names = target_names))

# Confustion matrix visualisation
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()