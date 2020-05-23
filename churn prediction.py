#!/usr/bin/env python
# coding: utf-8

# # Customer Churn
# "Churn Rate" is a business term describing the rate at which customers leave or cease paying for a product or service.Consequently, there's growing interest among companies to develop better churn-detection techniques, leading many to look to data mining and machine learning for new and creative approaches. This is a post about modeling customer churn using Python.

# # The DataSet¶
# The data set I'll be using is a longstanding telecom customer data set..
# 
# The data is straightforward. Each row represents a subscribing telephone customer. Each column contains customer attributes such as phone number, call minutes used during different times of day, charges incurred for services, lifetime account duration, and whether or not the customer is still a customer.

# In[106]:


import pandas as pd
churn_df = pd.read_csv('churn.csv.txt')


# In[105]:


churn_df.head()


# In[102]:


#From Sklearn library importing labelEncoder to convert categorical string columns into numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
churn_df['VMail Plan'] = le.fit_transform(churn_df['VMail Plan'].astype('str'))


# In[103]:


churn_df.head()


# In[74]:


churn_df['Churn?'] = le.fit_transform(churn_df['Churn?'].astype('str'))


# In[75]:


churn_df.head()


# In[76]:


churn_df['Int\'l Plan'] = le.fit_transform(churn_df['Int\'l Plan'].astype('str'))


# In[77]:


churn_df.head()


# In[78]:


churn_df.columns


# In[79]:


#df1 = churn_df.rename(columns={['Int\'l Plan': intl]})


# In[80]:


#churn_df1=pd.get_dummies(churn_df['Int\'l Plan'])


# In[81]:


#churn_df_new=pd.concat([churn_df1,churn_df],axis=1)
#churn_df_new.head()


# In[82]:


col = ['State','Area Code','Int\'l Plan','Phone']
churn_df = churn_df.drop(col,axis=1)
churn_df.head()


# In[83]:


churn_df.dtypes


# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
churn_df.plot(x = 'VMail Plan',y='Churn?', kind='scatter')


# In[85]:


churn_df.plot(x = 'VMail Message',y='Churn?', kind='scatter')


# In[86]:


churn_df.plot(x = 'CustServ Calls',y='Churn?', kind='scatter')


# In[87]:


churn_df.plot(x = 'CustServ Calls',y='Churn?', kind='hist')


# In[ ]:





# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


#Assigning target variable into Y
X=churn_df.iloc[:,0:15] 
Y=churn_df[['Churn?']]
X.head()


# In[90]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y)


# In[91]:


X_train.shape


# In[92]:


X_train.tail()


# In[93]:


Y_train.head()


# In[94]:


type(Y_train)


# In[95]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)


# In[96]:


knn.fit(X_train, Y_train)


# In[97]:


knn.score(X_test, Y_test)


# In[98]:


from sklearn import svm
clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)

# Now measure its performance with the test data
clf.score(X_test, Y_test)


# In[ ]:




