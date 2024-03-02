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

# Display the list of tables
print(tables_df)

# Example query with actual table name
query = "SELECT * FROM lung_cancer"

# Load data into a DataFrame
df = pd.read_sql_query(query, conn)

# Display the first few rows of the DataFrame
print(df.head())

# Close the connection
conn.close()

print(df.info())
print(df.describe())

# Drop duplicate rows 
df_clean = df.drop_duplicates()
print(df_clean.info())

# Missing values overview
print(df_clean.isna().sum())

# Drop missing values (5% or less of total values)
threshold = len(df) * 0.05 
cols_to_drop = df_clean.columns[df_clean.isna().sum() <= threshold]
df_clean = df_clean.dropna(subset = cols_to_drop)

print(df_clean.isna().sum())

# Convert inconsistent data types

# Convert 'ID' column to float
df_clean['ID'] = df_clean['ID'].astype('int64')

# Fix inconsistent data 

# Convert all strings to lowercase 
df_clean = df_clean.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

# Remove rows with 'nan' values from Gender column
df_clean = df_clean.drop(df_clean[df_clean['Gender'].str.contains('nan', na = False)].index)
print(df_clean['Gender'].unique())

print(df_clean.info())

# Impute missing values for 'COPD History' and 'Taken Bronchodilators' 
# NaN values are converted into 'no'

df_clean.fillna('no', inplace = True)
print(df_clean.info())

# Encode categorical values 

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
df_clean['Tiredness'] = df_clean['Frequency of Tiredness'].map(fot_mapping)

# Encoding 'Dominant Hand' column
df_clean['Right Handed'] = df_clean['Dominant Hand'].apply(lambda val:0 if val == 'left' else 1)

# Transforming numeric values 

# Transforming 'Last Weight' and 'Current Weight' into 'Weight Change' 
df_clean['Weight Change'] = df_clean['Current Weight'] - df_clean['Last Weight']

# Encoding 'Start Smoking' column to 'Smoker'
df_clean['Smoker'] = df_clean['Start Smoking'].apply(lambda val:0 if val == 'not applicable' else 1)

# Transforming 'still_smoking' in 'Stop Smoking' column into '2024'
# 2024 is most recent value in 'Stop Smoking' column
df_clean['Stop Smoking 2'] = df_clean['Stop Smoking'].apply(lambda val:2024 if val == 'still smoking' else val)

# Transforming 'not applicable' in 'Start Smoking' and 'Stop Smoking' to 0
df_clean['Stop Smoking 2'] = df_clean['Stop Smoking 2'].apply(lambda val:0 if val == 'not applicable' else val)
df_clean['Start Smoking 2'] = df_clean['Start Smoking'].apply(lambda val:0 if val == 'not applicable' else val)

# Transforming 'Start Smoking' and 'Stop Smoking' into 'Years Smoking'
df_clean['Start Smoking 2'] = df_clean['Start Smoking 2'].astype('int64')
df_clean['Stop Smoking 2'] = df_clean['Stop Smoking 2'].astype('int64')
df_clean['Years Smoking'] = df_clean['Stop Smoking 2'] - df_clean['Start Smoking 2']

# Handling outliers 

# 'Weight Change' column 
# Identifying 25th and 75th percentile 
weightchange_25 = df_clean['Weight Change'].quantile(0.25)
weightchange_75 = df_clean['Weight Change'].quantile(0.75)
weightchange_iqr = weightchange_75 - weightchange_25 

# Identifying thresholds 
wc_lower = weightchange_25 - (1.5 * weightchange_iqr)
wc_upper = weightchange_75 + (1.5 * weightchange_iqr)
print(wc_lower, wc_upper)

# 'Years Smoking' column
# Identifying 25th and 75th percentile 
yearssmoking_25 = df_clean['Years Smoking'].quantile(0.25)
yearssmoking_75 = df_clean['Years Smoking'].quantile(0.75)
yearssmoking_iqr = yearssmoking_75 - yearssmoking_25 

# Identifying thresholds 
ys_lower = yearssmoking_25 - (1.5 * yearssmoking_iqr)
ys_upper = yearssmoking_75 + (1.5 * yearssmoking_iqr)

print(ys_lower, ys_upper)
df_clean[(df_clean['Years Smoking'] < ys_lower) | 
         (df_clean['Years Smoking'] > ys_upper)][['ID', 'Age','Start Smoking', 
                                                  'Stop Smoking','Years Smoking',
                                                  'Lung Cancer Occurrence']]

# Handling outliers
# The dataset features rows where the 'Age' column is negative 
# This could be an input error, hence these values are changed to positive

df_clean['Age New'] = df_clean['Age'].apply(lambda val:-val if val <0 else val)

# 'Age New' column
# Identifying 25th and 75th percentile 
age_25 = df_clean['Age New'].quantile(0.25)
age_75 = df_clean['Age New'].quantile(0.75)
age_iqr = age_75 - age_25 

# Identifying thresholds 
age_lower = age_25 - (1.5 * age_iqr)
age_upper = age_75 + (1.5 * age_iqr)
print(age_lower, age_upper)

df_clean[(df_clean['Age New'] < age_lower) | 
         (df_clean['Age New'] > age_upper)][['ID', 'Age','Start Smoking', 
                                                  'Stop Smoking','Years Smoking',
                                                  'Lung Cancer Occurrence']]

print(df_clean)

df_clean.to_csv('data/cleaned_dataset.csv', index=False)