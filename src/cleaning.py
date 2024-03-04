import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Connect to the database
conn = sqlite3.connect('data/lung_cancer.db')

# Query to get a list of tables in the database
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"

# Execute the query and load the result into a DataFrame
tables_df = pd.read_sql_query(tables_query, conn)

# Query with actual table name
query = "SELECT * FROM lung_cancer"

# Load data into a DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Removing duplicate rows 
df_clean = df.drop_duplicates()

threshold = len(df) * 0.05 
cols_to_drop = df_clean.columns[df_clean.isna().sum() <= threshold]
df_clean = df_clean.dropna(subset = cols_to_drop)

df_clean['ID'] = df_clean['ID'].astype('int64')

# Convert all strings to lowercase 
df_clean = df_clean.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

# Remove rows with 'nan' string from 'Gender' column
df_clean = df_clean.drop(df_clean[df_clean['Gender'].str.contains('nan', na = False)].index)

# Imputing missing values for 'COPD History' and 'Taken Bronchodilators' 
# NaN values are converted into 'no'

df_clean.fillna('no', inplace = True)

# Encoding categorical values

# Encoding 'Gender' column to 'Male' 
df_clean['Male'] = df_clean['Gender'].apply(lambda val:1 if val == 'male' else 0)

# Encoding 'COPD History' column 
df_clean['COPDHistory'] = df_clean['COPD History'].apply(lambda val:1 if val == 'yes' else 0)

# Encoding 'Genetic Markers' column 
df_clean['GeneticMarkers'] = df_clean['Genetic Markers'].apply(lambda val:1 if val == 'present' else 0)

# Encoding 'Air Pollution Exposure' column 
ape_mapping = {'low':0, 'medium':1, 'high':2}
df_clean['AirPollutionExposure'] = df_clean['Air Pollution Exposure'].map(ape_mapping)

# Encoding 'Taken Bronchodilators' column 
df_clean['TakenBronchodilators'] = df_clean['Taken Bronchodilators'].apply(lambda val:1 if val == 'yes' else 0)

# Encoding 'Frequency of Tiredness' column 
fot_mapping = {'none / low':0, 'medium':1, 'high':2}
df_clean['FrequencyOfTiredness'] = df_clean['Frequency of Tiredness'].map(fot_mapping)

# Encoding 'Dominant Hand' column
df_clean['RightHanded'] = df_clean['Dominant Hand'].apply(lambda val:0 if val == 'left' else 1)

# Transforming numeric values 

# Transforming 'Last Weight' and 'Current Weight' into 'WeightChange' 
df_clean['WeightChange'] = df_clean['Current Weight'] - df_clean['Last Weight']

# Encoding 'Start Smoking' column to 'Smoker'
df_clean['Smoker'] = df_clean['Start Smoking'].apply(lambda val:0 if val == 'not applicable' else 1)

# Transforming 'still smoking' in 'Stop Smoking' column into '2024'
# 2024 is most recent value in 'Stop Smoking' column
df_clean['StopSmoking'] = df_clean['Stop Smoking'].apply(lambda val:2024 if val == 'still smoking' else val)

# Transforming 'not applicable' in 'Start Smoking' and 'Stop Smoking' to 0
df_clean['StopSmoking'] = df_clean['StopSmoking'].apply(lambda val:0 if val == 'not applicable' else val)
df_clean['StartSmoking'] = df_clean['Start Smoking'].apply(lambda val:0 if val == 'not applicable' else val)

# Transforming 'Start Smoking' and 'Stop Smoking' into 'Years Smoking'
df_clean['StartSmoking'] = df_clean['StartSmoking'].astype('int64')
df_clean['StopSmoking'] = df_clean['StopSmoking'].astype('int64')
df_clean['YearsSmoking'] = df_clean['StopSmoking'] - df_clean['StartSmoking']

# Handling outliers in 'Age'

# The dataset features rows where the 'Age' column is negative 
# This could be an input error, hence these values are changed to positive

df_clean['Age New'] = df_clean['Age'].apply(lambda val:-val if val <0 else val)

# Finalising data 

# Selecting columns from df_clean to keep
df_final = df_clean[['Age New', 'Male', 'COPDHistory', 'GeneticMarkers', 'AirPollutionExposure', 'WeightChange', 'Smoker', 
                     'YearsSmoking', 'TakenBronchodilators', 'FrequencyOfTiredness', 'RightHanded', 'Lung Cancer Occurrence']].copy()

# Renaming columns
df_final.rename(columns = {'Age New': 'Age', 'Lung Cancer Occurrence': 'LungCancerOccurrence'}, inplace = True)

df_final.to_csv('data/cleaned_dataset.csv', index=False)