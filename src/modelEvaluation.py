# Model Evaluation

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score, KFold, train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 

# Converting .csv into pandas df
df = pd.read_csv('data/cleaned_dataset.csv')

# Split dataset into features and target variable
X = df.drop(columns=['LungCancerOccurrence'])
y = df['LungCancerOccurrence']

# Split X and y into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling data 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

models = {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier(), 'Decision Tree': DecisionTreeClassifier()} 
results = [] 
for model in models.values():
     kf = KFold(n_splits = 6, random_state = 42, shuffle = True) 
     cv_results = cross_val_score(model, X_train_scaled, y_train, cv =  kf) 
     results.append(cv_results)

plt.boxplot(results, labels = models.keys()) 
plt.show()