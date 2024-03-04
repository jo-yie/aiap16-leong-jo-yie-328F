# k Nearest Neighbours
# Adapted from https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split

# Converting .csv into pandas df
df = pd.read_csv('data/cleaned_dataset.csv')

# Split dataset into features and target variable
X = df.drop(columns=['LungCancerOccurrence'])
y = df['LungCancerOccurrence']

# Adding Cross-Validation to improve accuracy
k_values = [i for i in range (1,31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))
    # print(k)
    # print(np.mean(score))

# Plotting K values against Accuracy Score
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
# plt.show()    

# Split X and y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify best K value 
best_index = np.argmax(scores)
best_k = k_values[best_index]

# Instantiate the model
knn = KNeighborsClassifier(n_neighbors = best_k)

# Fit model on the training data 
knn.fit(X_train, y_train)

# Predict target variable using test features
y_pred = knn.predict(X_test)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)