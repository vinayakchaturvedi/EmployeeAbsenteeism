# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:27:13 2019

@author: vinayak
"""

#Load libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import KNN
from scipy.stats import chi2_contingency

#Set working directory
os.chdir("C:/Users/vinayak\Desktop/EmployeeAbsenteesm")

#Load data
df = pd.read_excel("Absenteeism_at_work_Project.xls")
#Remove space and characters from column names
df.columns = df.columns.str.strip().str.replace('/',' per ').str.replace(' ','_')

################################## Exploratory Data Analysis ################################################
#Information about datatype of columns
df.info()
df['Reason_for_absence'] = df['Reason_for_absence'].replace(0, np.nan)

#Univariate Analysis and Variable Consolidation --> Transform into proper data type
cnumber_factor = [0,1,2,3,4,11,12,13,14,15,16]
for i in cnumber_factor:
    df.iloc[:,i] = df.iloc[:,i].astype('object')
    
df.info()            
        
#Categorical variable
cnames_factor = df.select_dtypes(include=['object']).columns
#Numeric variable
cnumber_numeric = [5,6,7,8,9,10,17,18,19,20]
cnames_numeric = df.select_dtypes(exclude=['object']).columns
#Remove target variable from cnames_numeric
cnames_numeric = cnames_numeric.drop('Absenteeism_time_in_hours')

##################################Missing value analysis################################################
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
    
missing_val = pd.DataFrame(df.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_count'})

#descending order
missing_val = missing_val.sort_values('Missing_count', ascending = False).reset_index(drop = True)

#save output results 
missing_val.to_csv("Missing_count.csv")

missing_val['Missing_count'].sum()
#There are 178 missing values are present in the dataset so we need to perform missing value analysis.

#KNN imputation
#Assigning levels to the categories
lis = []
for i in range(0, df.shape[1]):
    if(df.iloc[:,i].dtypes == 'object'):
        df.iloc[:,i] = pd.Categorical(df.iloc[:,i])
        df.iloc[:,i] = df.iloc[:,i].cat.codes 
        df.iloc[:,i] = df.iloc[:,i].astype('object')
        lis.append(df.columns[i])

#replace -1 with NA to impute
for i in range(0, df.shape[1]):
    df.iloc[:,i] = df.iloc[:,i].replace(-1, np.nan) 

#Apply KNN imputation algorithm
df = pd.DataFrame(KNN(k = 3) .fit_transform(df), columns = df.columns)

#Convert into proper datatypes
for i in lis:
    df.loc[:,i] = df.loc[:,i].round()
    df.loc[:,i] = df.loc[:,i].astype('object')

################################## Analyze Data Insights ##########################################

df[cnames_numeric].describe().to_csv("C:/Users/vinayak/Desktop/EmployeeAbsenteesm/ab.csv")
df[cnames_numeric].describe()

#Analyze Distribution
number_of_columns=9
number_of_rows = len(cnames_numeric)-1/number_of_columns
plt.figure(figsize=(5*number_of_columns,8*number_of_rows))
for i in range(0,len(cnames_numeric)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.distplot(df[cnames_numeric[i]],kde=True) 


##################################Outlier Analysis################################################
plt.figure(figsize=(number_of_columns,5*number_of_rows))
for i in range(0,len(cnames_numeric)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.set_style('whitegrid')
    sns.boxplot(df[cnames_numeric[i]],color='green',orient='v')
    plt.tight_layout()


#Detect and replace with NA
#Extract quartiles

for i in cnames_numeric:
    q75, q25 = np.percentile(df.loc[:,i], [75 ,25])
    #Calculate IQR
    iqr = q75 - q25
    #Calculate inner and outer fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    #Replace with NA
    df.loc[df.loc[:,i] < minimum,i] = np.nan
    df.loc[df.loc[:,i] > maximum,i] = np.nan
     
missing_val = pd.DataFrame(df.isnull().sum())

#Apply KNN imputation algorithm
df = pd.DataFrame(KNN(k = 3) .fit_transform(df), columns = df.columns)

#Convert into proper datatypes
for i in lis:
    df.loc[:,i] = df.loc[:,i].round()
    df.loc[:,i] = df.loc[:,i].astype('object')

df.loc[:,'Absenteeism_time_in_hours'] = df.loc[:,'Absenteeism_time_in_hours'].round()
##################################Feature Selection################################################
#Remove the variable that are not useful for the analysis

df_corr = df.loc[:,cnames_numeric]
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(10, 10))
#Generate correlation matrix
corr = df_corr.corr()
#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap = 'viridis',
            square=True, ax=ax, annot=True)

#As we can see in the plot weight and body mass index are very highly +ve correlated so we can remove 1 of them 
#And weight is less related to AbsenteeismHour as compared to body mass index so I am removing weight


for i in range(0,len(cnames_factor)):
    for j in range(i+1,len(cnames_factor)):
        print(cnames_factor[i], " VS ", cnames_factor[j])
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(df[cnames_factor[i]], df[cnames_factor[j]]))
        print(p)

#In the Ch-square test we can see that social smoker is dependent on almost all other independent variable
# So we can remove it
df = df.drop(["Social_smoker","Weight"], axis=1) 
cnames_numeric=cnames_numeric.drop('Weight')
cnames_factor = cnames_factor.drop('Social_smoker')
################################## Feature Scaling ################################################
#Normality check - Done in Analyze Data Insights :: Data is not normally distributed
# Apply normalization

#plt.hist(df['Transportation_expense'], bins='auto')

#Nomalisation
#for i in cnames_numeric:
  #  print(i)
  #  df[i] = (df[i] - min(df[i]))/(max(df[i]) - min(df[i]))

#No need to apply feature scaling as in our problem we need to identify the reason for absenteeism 
#nothing to predict here (Human Readable -- Actual Values)
  
################################## Result ################################################

#Que1: What changes company should bring to reduce the number of absenteeism?
count = pd.DataFrame(df['Absenteeism_time_in_hours'].value_counts()).sort_index()
numeric_impact = df.groupby('Absenteeism_time_in_hours')[cnames_numeric].mean()
count = count.reset_index()
numeric_impact = numeric_impact.reset_index()
count = count.rename(columns = {'index': 'Absenteeism_time_in_hours', 'Absenteeism_time_in_hours': 'Count'})

result1 = pd.merge(numeric_impact, count, on='Absenteeism_time_in_hours')

plt.figure(figsize=(15,6))
sns.barplot(data=df, x="Reason_for_absence", y="Absenteeism_time_in_hours")

plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Month_of_absence", y="Absenteeism_time_in_hours")

plt.figure(figsize=(7,4))
sns.barplot(data=df, x="Seasons", y="Absenteeism_time_in_hours")

plt.figure(figsize=(3,3))
sns.barplot(data=df, x="Social_drinker", y="Absenteeism_time_in_hours")

plt.figure(figsize=(6,3))
sns.barplot(data=df, x="Day_of_the_week", y="Absenteeism_time_in_hours")

#Que2:  How much losses every month can we project in 2011 if same trend of absenteeism continues? 
result2 = df.groupby('Month_of_absence')['Absenteeism_time_in_hours'].mean()
result2 = result2.reset_index()
