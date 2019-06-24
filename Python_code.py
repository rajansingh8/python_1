import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('', index_col = 0)
test_data = pd.read_csv('', index_col = 0)

#lets check the shape and head of the data
print("shape of train", train_data.shape)
print("Shape of Test :", test_data.shape)

# lets look at the head of the train & test

train_data.head()
test_data.head()

# let's decsribe the train & test set

train_data.describe()
test_data.decsribe()

# Let get the info of train & test

train_data.info()
test_data.info()

# Let get the data types of train & test

train_data.dtypes
test_data.dtypes

# Checking if there exists any NULL values in the train & test set

train_data.isnull().sum()
test_data.isnull().sum()

# Let check the values present in the Employement.Type attribute in the train and test sets

train_data['Employment.Type'].value_counts()

# Let fill unemployed in the place of Null values

train_data['Employment.Type'].fillna('Unemployed', inplace = True)
test_data['Employment.Type'].fillna('Unemployed', inplace = True)

# let's check if there is any null values still left or not
print("Null values left in the train set:", train_data.isnull().sum().sum())
print("Null values left in the test set:", test_data.isnull().sum().sum())

# Let save the unique id of the test set and labels set

unique_id = test_data['UniqueID']
y = train_data['loan_default']

# let's delete the last column from the dataset to  concat train and test
train_data = train_data.drop(['loan_default'], axis = 1)

# shape of train
train_data.shape, test_data.shape,y.shape

# lets concat the train and test sets for preprocessing and visualizations

r_data = pd.concat([train_data, test_data], axis = 0,ignore_index=True)

# let's check the shape
r_data.shape

r_data['Employment.Type'].value_counts()

# Lets plot a donut chart

size = [187429, 147013, 11104]
colors = ['pink', 'lightblue', 'lightgreen']
labels = "Self Employed", "Salaried", "Unemployed" 
explode = [0.05, 0.05, 0.05]

circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, pctdistance = 1, autopct = '%.2f%%')
plt.title('Types of Employments', fontsize = 30)
plt.axis('off')
p = plt.gcf()
p.gca().add_artist(circle)
plt.legend()
plt.show()

# encodings for type of employments

r_data['Employment.Type'] = r_data['Employment.Type'].replace(('Self employed', 'Salaried', 'Unemployed'), (2, 1, 0))
train_data['Employment.Type'] = train_data['Employment.Type'].replace(('Self employed', 'Salaried', 'Unemployed'), (2, 1, 0))
test_data['Employment.Type'] = test_data['Employment.Type'].replace(('Self employed', 'Salaried', 'Unemployed'), (2, 1, 0))

# checking the values  of employement type
r_data['Employment.Type'].value_counts()

r_data.columns

#let's check the unique values of ids in different branchs

print("Total no. of Unique Ids :", r_data['UniqueID'].nunique())
print("Total no. of Unique Branches :", r_data['branch_id'].nunique())
print("Total no. of Unique Suppliers :", r_data['supplier_id'].nunique())
print("Total no. of Unique Manufactures :", r_data['manufacturer_id'].nunique())
print("Total no. of Unique Current pincode Ids :", r_data['Current_pincode_ID'].nunique())
print("Total no. of Unique State IDs :", r_data['State_ID'].nunique())
print("Total no. of Unique Employee code IDs :", r_data['Employee_code_ID'].nunique())

# check the distribution of disbursed amount

plt.rcParams['figure.figsize'] = (18, 5)

plt.subplot(1, 3, 1)
sns.distplot(data['disbursed_amount'],  color = 'orange')
plt.title('Disburesed Amount')

plt.subplot(1, 3, 2)
sns.distplot(r_data['asset_cost'], color = 'pink')
plt.title('Asset Cost')

plt.subplot(1, 3, 3)
sns.distplot(r_data['ltv'], color = 'red')
plt.title('Loan to value of the asset')

plt.show()

#performing log transformations on disbursed amount, ltv, and asset cost

r_data['disbursed_amount'] = np.log1p(r_data['disbursed_amount'])
r_data['ltv'] = np.log1p(r_data['ltv'])
r_data['asset_cost'] = np.log1p(r_data['asset_cost'])

train_data['disbursed_amount'] = np.log1p(train_data['disbursed_amount'])
train_data['ltv'] = np.log1p(train_data['ltv'])
train_data['asset_cost'] = np.log1p(train_data['asset_cost'])

test_data['disbursed_amount'] = np.log1p(test_data['disbursed_amount'])
test_data['ltv'] = np.log1p(test_data['ltv'])
test_data['asset_cost'] = np.log1p(test_data['asset_cost'])

plt.rcParams['figure.figsize'] = (18, 5)

plt.subplot(1, 3, 1)
sns.distplot(r_data['disbursed_amount'],  color = 'orange')
plt.title('Disburesed Amount')

plt.subplot(1, 3, 2)
sns.distplot(r_data['asset_cost'], color = 'pink')
plt.title('Asset Cost')

plt.subplot(1, 3, 3)
sns.distplot(r_data['ltv'], color = 'red')
plt.title('Loan to value of the asset')

plt.show()

# let's first convert the date into date-time format

r_data['Date.of.Birth'] = pd.to_datetime(r_data['Date.of.Birth'], errors = 'coerce')
train_r_data['Date.of.Birth'] = pd.to_datetime(train_r_data['Date.of.Birth'], errors = 'coerce')
test_r_data['Date.of.Birth'] = pd.to_datetime(test_r_data['Date.of.Birth'], errors = 'coerce')

# Extracting the year of birth of the customers
r_data['Year_of_birth'] = r_data['Date.of.Birth'].dt.year
train_r_data['Year_of_birth'] = train_r_data['Date.of.Birth'].dt.year
test_r_data['Year_of_birth'] = test_r_data['Date.of.Birth'].dt.year
r_data['Year_of_birth'].min(), r_data['Year_of_birth'].max()

r_data.loc[r_data['Date.of.Birth'].dt.year >2001,'Date.of.Birth'] = pd.to_datetime(20010101,format='%Y%m%d')
train_data.loc[train_data['Date.of.Birth'].dt.year >2001,'Date.of.Birth'] = pd.to_datetime(20010101,format='%Y%m%d')
test_data.loc[train_data['Date.of.Birth'].dt.year >2001,'Date.of.Birth'] = pd.to_datetime(20010101,format='%Y%m%d')

import datetime as DT
now = pd.Timestamp(DT.datetime.now())
r_data['age'] = (now - r_data['Date.of.Birth']).astype('<m8[Y]') 
r_data['Year_of_birth'] = r_data['Date.of.Birth'].dt.year

train_data['age'] = (now - train_data['Date.of.Birth']).astype('<m8[Y]') 
train_data['Year_of_birth'] = train_data['Date.of.Birth'].dt.year

test_data['age'] = (now - test_data['Date.of.Birth']).astype('<m8[Y]') 
test_data['Year_of_birth'] = test_data['Date.of.Birth'].dt.year
sns.distplot(r_data['age'],color='red')
plt.title('Distribution by age')

# checking the values inside date of year
sns.distplot(r_data['Year_of_birth'], color = 'blue')
plt.title('Distribution of Year of birth')




