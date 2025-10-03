#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 11:07:45 2025

@author: wujiayi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- Load and preprocess dataset ---
file_path = "/Users/wujiayi/Desktop/FTD/AML/credit_risk_dataset.csv"
df = pd.read_csv(file_path)
if 'loan_grade' in df.columns:
    df = df.drop(columns=['loan_grade'])

feature_rename_dict = {
    'loan_percent_income': 'Percentage of loan to income',
    'loan_int_rate': 'Loan interest rate',
    'person_age': 'Age',
    'person_income': 'Income',
    'loan_amnt': 'Amount of the loan',
    'person_emp_length': 'Employment length',
    'cb_person_cred_hist_length': 'Credit history length'
}

numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                  'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']


#original_rows = len(df)
#df = df[df['person_emp_length'] <= df['person_age']]
#rows_removed = original_rows - len(df)
#print(f"Rows removed: {rows_removed}")
df = df[df['person_age'] <= 100]
df = df[df['person_emp_length'] <= df['person_age']]

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f"{feature_rename_dict.get(col, col)}")
    plt.xlabel(feature_rename_dict.get(col, col))
    plt.ylabel("Frequency")
    plt.tight_layout()
    #plt.savefig(f"/Users/wujiayi/Desktop/FTD/AML/graphs/{col}_histogram_before.png", dpi=300)
    plt.close()

# Plot boxplots separately
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col], vert=False)
    plt.title(f"{feature_rename_dict.get(col, col)}")
    plt.xlabel(feature_rename_dict.get(col, col))
    plt.tight_layout()
    #plt.savefig(f"/Users/wujiayi/Desktop/FTD/AML/graphs/{col}_boxplot_before.png", dpi=300)
    plt.show()


#%% Why drop loan_grade

df_loan_grade = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_grade', y='loan_int_rate', data=df_loan_grade, order=sorted(df_loan_grade['loan_grade'].unique()))
plt.title('Loan Interest Rate by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Interest Rate (%)')
plt.savefig("/Users/wujiayi/Desktop/FTD/AML/graphs/Loan_Interest_Rate_by_Loan_Grade.png", dpi=300)
plt.show()

### Compute correlation
grade_map = {grade: idx for idx, grade in enumerate(sorted(df_loan_grade['loan_grade'].unique()))}
df_loan_grade['loan_grade_encoded'] = df_loan_grade['loan_grade'].map(grade_map)

correlation = df_loan_grade[['loan_grade_encoded', 'loan_int_rate']].corr().iloc[0, 1]
print(f"Pearson correlation between loan_grade and loan_int_rate: {correlation:.2f}")

#%%

# Loop over each column to create the combined plots
for col in numerical_cols:
    data = df[col].dropna()
    
    # Calculate quartiles and IQR
    Q1 = data.quantile(0.25)
    Q2 = data.median()
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR

    # Set up plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"Distribution and Boxplot of {feature_rename_dict.get(col, col)}", fontsize=14)

    # Histogram with quartile lines
    sns.histplot(data, kde=False, ax=ax1, bins=30)
    for q, label in zip([Q1, Q2, Q3], ['Q1', 'Q2', 'Q3']):
        ax1.axvline(q, color='red', linestyle='--')
        ax1.text(q, ax1.get_ylim()[1]*0.9, label, rotation=90, color='red')

    # Boxplot
    sns.boxplot(x=data, ax=ax2, orient='h')
    ax2.axvline(lower_whisker, color='gray', linestyle=':', label='Lower whisker')
    ax2.axvline(upper_whisker, color='gray', linestyle=':', label='Upper whisker')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    #plt.savefig(f"/Users/wujiayi/Desktop/FTD/AML/graphs/{col}_after.png", dpi=300)
    plt.show()

