# Linear Regression
# Adapted from https://realpython.com/linear-regression-in-python/

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/cleaned_dataset.csv')

x = df.drop(columns=['Lung Cancer Occurrence'])
y = df['Lung Cancer Occurrence']

# Add a train/test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)