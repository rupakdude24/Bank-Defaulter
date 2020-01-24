#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading required libraries
import pandas as pd 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats 
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Setting Path to directory
os.getcwd()
os.chdir("C:/Users/Rupak/Desktop/edwisor/Project 2")
os.getcwd()


# In[49]:


#Fetching csv file and checking
bank_loan=pd.read_csv("bank-loan.csv")
bank_loan.head()


# In[4]:


#bike.csv statistics metrics
bank_loan.describe()


# In[5]:


#bike.csv datatypes
bank_loan.dtypes


# In[6]:


#Missing values
bank_loan.isnull().sum()


# In[50]:


#Replacing some variables with some meaningful values
bank_loan['default'] = bank_loan['default'].replace([1,0],["Defaulter","Not defaulter"])


# In[51]:


#Converting into Categorical type
bank_loan['ed'] = bank_loan['ed'].astype('category')
bank_loan['default'] = bank_loan['default'].astype('category')


# In[9]:


#Using Boxplot for outliers
sns.boxplot(data=bank_loan[['employ','address','age','creddebt','othdebt','debtinc']])
fig=plt.gcf()
fig.set_size_inches(8,8)


# In[10]:


#Using Boxplot for outliers
sns.boxplot(data=bank_loan[['income']])
fig=plt.gcf()
fig.set_size_inches(8,8)


# In[52]:


#Making copy of original csv
bank = bank_loan.copy()
bank.head()


# In[82]:


#Removing outliers in debtinc using IQR
q75, q25 = np.percentile(bank['debtinc'], [75 ,25])
print(q75,q25)
iqr = q75 - q25
print(iqr)
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
print(min)
print(max)


# In[63]:


#Displaying outliers in debtinc
bank[(bank.debtinc < min) | (bank.debtinc > max)]


# In[83]:


#Removing outliers from debtinc
#bank.drop(bank[(bank.debtinc > max) | (bank.debtinc < min) ].index , inplace=True)


# In[90]:


#Removing outliers in creddebt using IQR
q75, q25 = np.percentile(bank['creddebt'], [75 ,25])
print(q75,q25)
iqr = q75 - q25
print(iqr)
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
print(min)
print(max)


# In[85]:


#Displaying outliers in creddebt
bank[(bank.creddebt < min) | (bank.creddebt > max)]


# In[86]:


#Removing outliers from creddebt
#bank.drop(bank[(bank.creddebt > max) | (bank.creddebt < min) ].index , inplace=True)


# In[53]:


bank = bank.dropna()


# In[96]:


#Checking the dimensions after removing outliers
bank.shape


# In[20]:


#Seeing effect on some variables after dropping missing values
sns.boxplot(data=bank[['creddebt','debtinc']])
fig=plt.gcf()
fig.set_size_inches(8,8)


# In[21]:


#Exploring some categorical variable
sns.set_style("whitegrid")
sns.factorplot(data=bank, x='ed', kind= 'count',size=4,aspect=2)


# In[22]:


#Distribution of numerical data using histogram
plt.hist(data=bank, x='employ', bins='auto', label='Temperature')
plt.xlabel('Employment status(converted in numeric format)')
plt.title("Job Status Distribution")


# In[23]:


#Distribution of numerical data using histogram
plt.hist(data=bank, x='address', bins='auto', label='address')
plt.xlabel('Geographic area(converted in numeric format)')
plt.title("Geographic area Distribution")


# In[84]:


#Distribution of numerical data using histogram
bank.plot.scatter(x='income', y='debtinc', title= "Scatter plot between income & debtinc",color='red');
bank.plot.scatter(x='income', y='address', title= "Scatter plot between income & address",color='red');


# In[10]:


#Correlation matrix
bank.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[54]:


#preparing dataset for modeling
cols1 = ['age','employ','address','income','debtinc','creddebt','othdebt','ed']
X=bank[cols1]
y=bank['default']


# In[56]:


#Splitting Data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[57]:


#Logistic regression
#Accuracy = 78.57%
#Recall/Sensitivity/true pos rate = 38.4%
#Specificity/True neg rate = 94.05%
#Precision = 71.4%
logreg = LogisticRegression()
logreg.fit(x_train, y_train)


# In[58]:


#bank['default'] = bank['default'].replace(["Defaulter","Not defaulter"],[1,0])


# In[59]:


#Prediction
from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(x_test)


# In[60]:


#Confusion matrix for Logistic regression
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[18]:


#Accuracy of Logistic regression classifier: 78%
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))


# In[19]:


# Precision , recall for Logistic regression classifier
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[64]:


#Preparing data for Random forest algorithm by label encoding of Categorical Variables 
#Importing Library for label encoding
from sklearn.preprocessing import LabelEncoder

train_data, test_data = train_test_split(bank, test_size=0.2)

test_data['default'] = test_data['default'].cat.codes
train_data['default'] = train_data['default'].cat.codes

test_data['ed'] = test_data['ed'].cat.codes
train_data['ed'] = train_data['ed'].cat.codes


# In[66]:


train_predictor1 = train_data[['age','employ','address','income','debtinc','creddebt','othdebt','ed']].values
train_target1 = train_data['default'].values

test_predictor1 = test_data[['age','employ','address','income','debtinc','creddebt','othdebt','ed']].values
test_target1 = test_data['default'].values


# In[71]:


#Random Forest Classifier
#Accuracy = 83.57%
#Recall/Sensitivity/true pos rate = 53.33%
#Specificity/True neg rate = 91.81%
#Precision = 64%
from sklearn.ensemble import RandomForestClassifier
randome_one = RandomForestClassifier(n_estimators= 500,bootstrap = True, random_state=100).fit(train_predictor1, train_target1)


# In[72]:


#Prediction by random forest classifier
random_predict= randome_one.predict(test_predictor1)


# In[74]:


#Prediction probabilities
rf_probs = randome_one.predict_proba(test_predictor1)[:, 1]


# In[76]:


#Confusion matrix of random forest classifier
cnf_matrix = metrics.confusion_matrix(test_target1, random_predict)
cnf_matrix


# In[75]:


#Precsion, recall for random forest classifier
print(classification_report(test_target1, random_predict))


# In[79]:


#from sklearn.metrics import roc_auc_score

# Calculate roc auc
#roc_value = roc_auc_score(test_target1, rf_probs)
#roc_value

