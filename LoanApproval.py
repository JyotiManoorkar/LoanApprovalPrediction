#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[10]:


df=pd.read_excel('C:\JyotiLearnings\GitDemo\MachineLearning_loanApprovalPredictionSystemUsingPython\loan.xlsx')


# In[11]:


df.head()


# In[12]:


df.info()


# In[57]:


df['ApplicantIncome'] = df['ApplicantIncome'].astype(int) 
df.isnull().sum() #find missing values- total number of missing data


# In[58]:


df['loanAmount_log']=np.log(df['LoanAmount']) #find natual log
df['loanAmount_log'].hist(bins=20)


# In[59]:


df.isnull().sum()


# In[60]:


df['Total_Income']=df['ApplicantIncome']+df['CoapplicantIncome'] 
df['Total_Income']=np.log(df['Total_Income']) #find natual log
df['Total_Income'].hist(bins=20)


# In[61]:


df.isnull().sum()


# In[62]:


# fill null values in respective columns

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True) 
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log=df.loanAmount_log.fillna(df.loanAmount_log.mean())


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.isnull().sum()


# In[63]:


x=df.iloc[:,np.r_[1:5,9:11,13:15]].values #Select specific rows and columns for trainig and testing
y=df.iloc[:,12].values


# In[64]:


x


# In[65]:


y


# In[66]:


# find percentage missing gender

print("per of missing gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[67]:


## find number of perople who take loan as group by gender
print("number of perople who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df,palette='Set1')


# In[68]:


## find number of perople who take loan as group by Marital_Status
print("number of perople who take loan as group by Marital_Status:")
print(df['Married'].value_counts())
sns.countplot(x='Married',data=df,palette='Set1')


# In[69]:


## find number of perople who take loan as group by dependents
print("number of perople who take loan as group by dependents:")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents',data=df,palette='Set1')


# In[70]:


## find number of perople who take loan as group by Self_Employed
print("number of perople who take loan as group by Self_Employed:")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=df,palette='Set1')


# In[71]:


## find number of perople who take loan as group by LoanAmount
print("number of perople who take loan as group by LoanAmount:")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=df,palette='Set1')


# In[72]:


## find number of perople who take loan as group by Credithistory
print("number of perople who take loan as group by Credithistory:")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=df,palette='Set1')


# In[73]:


# import scikit learn library from training and testing

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import LabelEncoder
Labelencoder_x=LabelEncoder()


# In[75]:


for i in range(0,5): 
    X_train[:,i]= X_train[:,i].astype(str)
    X_train[:,i]= Labelencoder_x.fit_transform(X_train[:,i])
    X_train[:,7]= X_train[:,7].astype(str)
    X_train[:,7]= Labelencoder_x.fit_transform(X_train[:,7])
X_train


# In[77]:


Labelencoder_y=LabelEncoder()
y_train=Labelencoder_y.fit_transform(y_train)
y_train        # Training is done


# In[80]:


for i in range(0,5):
    X_test[:,i]= X_test[:,i].astype(str)
    X_test[:,i]= Labelencoder_x.fit_transform(X_test[:,i])
    X_test[:,7]= X_test[:,7].astype(str)
    X_test[:,7]= Labelencoder_x.fit_transform(X_test[:,7])
X_train


# In[81]:


Labelencoder_y=LabelEncoder()
y_test=Labelencoder_y.fit_transform(y_test)
y_test   


# In[91]:


from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
x_test=ss.fit_transform(X_test)


# In[92]:


from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier()
rf_clf.fit(X_train,y_train)


# In[95]:


from sklearn import metrics
y_pred=rf_clf.predict(x_test)
print("Accuracy of RandomForestClassifier is",metrics.accuracy_score(y_pred,y_test))
y_pred


# In[96]:


from sklearn.naive_bayes import GaussianNB
nb_clf=GaussianNB()
nb_clf.fit(X_train,y_train)


# In[100]:


y_pred=nb_clf.predict(X_test)
print("Accuracy of Naive Bayes Classifier is",metrics.accuracy_score(y_pred,y_test))
y_pred


# In[98]:


from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)


# In[105]:


y_pred=dt_clf.predict(X_test)
print("Accuracy of Decision Tree Classifier is",metrics.accuracy_score(y_pred,y_test))
y_pred


# In[107]:


from sklearn.neighbors import KNeighborsClassifier
kn_clf=KNeighborsClassifier()
kn_clf.fit(X_train,y_train)


# In[108]:


y_pred=kn_clf.predict(X_test)
print("Accuracy of KNeighbors Classifier is",metrics.accuracy_score(y_pred,y_test))
y_pred


# In[ ]:




