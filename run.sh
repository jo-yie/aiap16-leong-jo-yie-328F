#!/bin/bash
echo "Cleaning and preprocessing dataset..."
python src/cleaning.py

echo "Implementing decision tree algorithm..."
python src/decisionTrees.py

echo "Implementing kNN algorithm..."
python src/knn.py

echo "Implementing logistic regression algorithm..."
python src/logisticRegression.py

echo "Comparing implemented algorithms and generating graph..."
python src/modelEvaluation.py