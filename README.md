# aiap16-leong-jo-yie-328F
# Leong Jo Yie 
# joyieleong@gmail.com

## Source Code (src) 
The 'src' directory contains the source code of the project, organised into modules for different tasks. 

## Files 
- **cleaning.py**: This module provides functions for data cleaning, preprocessing, and feature engineering based on the EDA performed. 
- **decisionTrees.py**: This module implements decision tree algorithms for classification.
- **knn.py**: This module contains the implementation of the k-nearest neighbours algorithm for classification.
- **logisticRegression.py**: This module implements logistic regression algorithms for classification tasks. 
- **modelEvaluation.py**: This module provides functions for evaluating and visualising the performance of all three machine learning models.

## Overview of Key Findings for Task 1 EDA 
In task 1, EDA was performed to understand the relationships between different variables in the dataset. Data cleaning and preprocessing was also undertaken to create a dataset which could easily be fed into a machine learning model. This included feature engineering for both numeric and categorical variables.

## Features in the Dataset 
The following table shows how features in the dataset were processed before being used in the machine learning models. 
| Feature Name         | Description                              | Processing Steps                                                          |
|----------------------|------------------------------------------|---------------------------------------------------------------------------|
| Age                  | Age of individuals in years              | - Negative values converted to positive                                   |
| Male                 | Gender of individuals                    | - Categorical encoding (0: Female, 1: Male)                               |
| COPDHistory          | Presence of COPD history                 | - Categorical encoding (0: No, 1: Yes)                                    |
| GeneticMarkers       | Presence of genetic markers              | - Categorical encoding (0: Not Present, 1: Present)                       |
| AirPollutionExposure | Level of air pollution exposure          | - Categorical encoding (0: None/Low, 1: Medium, 2: High)                  |
| WeightChange         | Change in individuals weight             | - Feature engineering (Current Weight - Last Weight)                      |
| Smoker               | Indicates whether individual is smoker   | - Feature engineering, categorical encoding (0: Not a smoker, 1: Smoker)  |
| YearsSmoking         | Number of years individual smoked        | - Feature engineering (Stop Smoking - Start Smoking)                      |
| TakenBronchodilators | History of taking bronchodilators        | - Categorical encoding (0: No, 1: Yes)                                    |
| FrequencyOfTiredness | Frequency of feeling tired               | - Categorical encoding (0: None/Low, 1: Medium, 2: High)                  |
| RightHanded          | Dominant hand                            | - Categorical encoding (0: Left Handed, 1: Right Handed)                  |
| LungCancerOccurrence | Diagnosed with lung cancer               | - Target variable for classification model                                |

## Machine Learning Models Used

**Decision Trees (decisionTrees.py)** 
Decision trees were selected as they can handle both numeric and categorical data, making them suitable for a medical dataset (like this one) that has a mix of data types. This model optimises decision tree performance through pre-pruning and setting a maximum depth for the DecisionTreeClassifier. The accuracy score before and after pre-pruning is outputted to show the effects of fine tuning the model. 

**K-Nearest Neighbour (knn.py)** 
KNN was selected as it is able to handle non-linear relationships, making it a valuable tool in a medical prediction task like this one. This model uses cross-validation to find the best k value with the highest accuracy score through looping through a range of values. This k value is then used to instantiate the KNN model and accuracy, precision, and recall are outputted to identify the effectiveness of the model. 

**Logistic Regression (logisticRegression.py)** 
Logistic regression was selected due to its applicability to binary classification tasks, hence it is well suited to predicting the presence or absence of lung cancer. The evaluation of this model outputs a confusion matrix, confusion matrix heatmap visualisation, and classification report to identify the relationships between true positives, false positives, false negatives, and true negatives. The classification report further outputs the precision, recall, and accuracy of the model. 

## High Level Evaluation of Models 
**modelEvaluation.py**
This module was used to give a high level overview of the accuracy of all three models in relation to one another. The output boxplot reveals that all three models have similar accuracy scores between 0.65 and 0.7. However, the KNN model has the highest median score, suggesting the KNN model is best-suited for predicting lung cancer occurrence in individuals. Hence it would be wise to further fine tune the KNN model to improve its accuracy. 