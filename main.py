import pandas as pd
import numpy as np

df_taxonomy = pd.read_csv('insurance_taxonomy.csv')
df_companies = pd.read_csv('ml_insurance_challenge.csv')

'''
Initial Data Exploration
'''

print('\nCompany shape: ' + str(df_companies.shape))
print('Taxonomy shape: ' + str(df_taxonomy.shape))

print('\nCompany Info:')
print(df_companies.info())
print('\nTaxonomy Info:')
print(df_taxonomy.info())

print('\nCompany First rows:')
print(df_companies.head())
print('\nTaxonomy First rows:')
print(df_taxonomy.head())

print('\nCompany Duplicates: ' + str(df_companies.duplicated().sum()))
print('Taxonomy Duplicates: ' + str(df_taxonomy.duplicated().sum()))


'''
Preprocessing
'''

# Fill NaN values
for col in ['description', 'sector', 'category']:
    df_companies[col].fillna('', inplace=True)
print('\nProcessed Company Info:')
print(df_companies.info())

# Drop duplicates
df_companies.drop_duplicates(inplace=True)
print('\nProcessed Company Duplicates: ' + str(df_companies.duplicated().sum()))


# Get column names of df_companies and combine them into a new column 'complete_description'
company_cols = df_companies.columns.tolist()
# print('\nCompany Columns: ' + str(company_cols))
df_companies['complete_description'] = df_companies[company_cols].agg(' '.join, axis=1)


# standardize text in 'complete_description'
df_companies['complete_description'] = df_companies['complete_description'].str.lower()
df_companies['complete_description'] = df_companies['complete_description'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
df_companies['complete_description'] = df_companies['complete_description'].str.replace(r'\s+', ' ', regex=True)
df_companies['complete_description'] = df_companies['complete_description'].str.strip() 


# display full text of first 5 rows of 'complete_description'
with pd.option_context('display.max_colwidth', None):
    print(df_companies['complete_description'].head(3).to_string(index=False))

# standardize text in 'label' of df_taxonomy
df_taxonomy['label'] = df_taxonomy['label'].str.lower()
print(df_taxonomy.head())