# Titanic Analysis
## Table of Content
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Expoloratory Data](#exploratory-data)
- [Data Analysis](#data-analysis)
- [Results](#results)
### Project Overview
This project analyzes Titanic passenger data, including information such as age, sex, passenger class, fare, number of siblings/spouses, number of parents/children, embarkation port, and survival status. It explores relationships between these variables and the likelihood of surviving the disaster.
### Data Sources
Dataset : https://www.kaggle.com/datasets/brendan45774/test-file
### Tools
- Phyton: 
1. Data Cleaning
2.  Exploring Data
### Exploratory Data
1. What is the distribution of survival status among passengers?
2. How is age distributed among passengers who survived and those who didn't?
3. How are survivors distributed across passenger classes and embarked ports?
4. How does the age distribution vary across different passenger classes for survivors?
5. Is there any correlation between the fare paid and the likelihood of survival?
6. How does the number of family members aboard affect the survival rate?7. How do the survival rates and number of family members vary across different passenger classes?
### Data Analysis
```phyton

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns

df = pd.read_csv('/kaggle/input/test-file/tested.csv')

print(df.head())
print(df.info())

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
duplicates_count = df.duplicated(subset=['PassengerId']).sum()

print("Count of duplicates in PassengerId:", duplicates_count)

df_cleaned = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

print(df_cleaned.head())

df_cleaned.describe()

import matplotlib.pyplot as plt
survived = df_cleaned[df_cleaned['Survived'] == 1]
not_survived = df_cleaned[df_cleaned['Survived'] == 0]
survived_count = df_cleaned['Survived'].value_counts()
plt.bar(['Not Survived', 'Survived'], survived_count, color=['red', 'green'])
plt.xlabel('Survival Status')
plt.ylabel('Number of Passengers')
plt.title('Distribution of Survival Status')
plt.show()
```

![distribution_of_survival_status](https://github.com/RomaDataAnalyst/TitanicDataAnalysis/assets/167080940/8dc9e772-825d-438e-b7ee-4df0300b6f9e)

```phyton
df_cleaned['Age'] = df_cleaned['Age'].fillna(df_cleaned['Age'].mean())
plt.hist([survived['Age'], not_survived['Age']], bins=20, color=['green', 'red'], alpha=0.5, label=['Survived', 'Not Survived'], edgecolor='black', histtype='barstacked')


plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age by Survival Status')
plt.legend()


plt.show()
```
![distribution_of_age_by_sruvival_status](https://github.com/RomaDataAnalyst/TitanicDataAnalysis/assets/167080940/d1724367-e407-4278-8d2f-136cd74ecd79)

```phyton
survived_data = df_cleaned[df_cleaned['Survived'] == 1]

sns.countplot(x='Pclass', data=survived_data, hue='Embarked')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Survivors')
plt.title('Distribution of Survivors by Passenger Class')
custom_labels = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
plt.legend(title='Embarked', labels=[custom_labels[label] for label in survived_data['Embarked'].unique()])


plt.show()
```
![survivors_by_passenger_class](https://github.com/RomaDataAnalyst/TitanicDataAnalysis/assets/167080940/c06a1dd6-0206-412b-bceb-6b802966ed75)
```phyton
sns.boxplot(x='Pclass', y='Age', data=survived_data)

plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.title('Distribution of Age for Survivors by Passenger Class')


plt.show()
```
![distribution_of_age_survivalstatus](https://github.com/RomaDataAnalyst/TitanicDataAnalysis/assets/167080940/aadd3157-2d3d-441f-b08b-25281f392fcc)

```phyton
sns.scatterplot(x='Fare', y='Survived', data=df_cleaned, hue='Survived', palette={0: 'red', 1: 'green'})
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.title('Survival Status by Fare')
plt.show()
```
![survival_status_by_fare](https://github.com/RomaDataAnalyst/TitanicDataAnalysis/assets/167080940/188a5e83-b785-458b-8e38-55fd9aa6a489)

```phyton
pearson_corr = df_cleaned['Fare'].corr(df_cleaned['Survived'], method='pearson')
print(f"Pearson correlation coefficient between Fare and Survived: {pearson_corr}")
```
Pearson correlation coefficient between Fare and Survived: 0.19151374269353372
```phyton
family_members = df_cleaned['Parch'] + df_cleaned['SibSp']
print(family_members.head())
sns.countplot(x=family_members, hue='Survived', data=df_cleaned,palette={0: 'red', 1: 'green'} )

plt.xlabel('Number of Family Members')
plt.ylabel('Count')
plt.title('Survival Status by Number of Family Members')

plt.show()
```
![survival_status_by_number_of_family_members](https://github.com/RomaDataAnalyst/TitanicDataAnalysis/assets/167080940/afd589e9-50c3-49a6-b6c8-60aeb01c7f39)

```phyton
sns.countplot(x='Pclass', hue=family_members, data=survived_data, palette='coolwarm')

plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.title('Survival Status and Family Members by Passenger Class')
plt.show()
```

![family_members_passenger_class](https://github.com/RomaDataAnalyst/TitanicDataAnalysis/assets/167080940/121db99b-6bc0-42be-99f9-b49623cf0d39)


### Results
The Titanic disaster resulted in a significant loss of life, especially among passengers aged 15-30. However, since this age group also constituted the largest portion of passengers, the impact might not be as straightforward.

Passengers from Southampton and Queenstown who traveled third class had better survival rates. This suggests socio-economic factors played a role in survival chances.

Average ages varied among classes, with first class passengers being older on average compared to those in second and third class.

Interestingly, there was a positive correlation between ticket price and survival, albeit not very strong. This implies that passengers who paid more for their tickets had slightly higher chances of survival, possibly due to better access to life-saving resources.

Family size also influenced survival rates. Passengers with 2, 3, or 5 family members had higher survival rates, except in second class where no one had more than 3. Only third class had passengers with over 5 family members.

These findings demonstrate that survival on the Titanic depended on a combination of factors including age, socio-economic status, ticket price, and family size. It was a complex scenario where multiple factors intertwined to determine survival outcomes.
