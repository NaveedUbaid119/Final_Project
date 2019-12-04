#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[2]:


da=pd.read_csv(r'/Users/IAmNaveed/Desktop/Bank-Marketing-Analysis-master/Data/small_ohe.csv')


# In[3]:


da.head()


# In[ ]:


#Splitting the Outcome and Input Variables


# In[5]:


X = da.drop('y', axis=1).values
y = da['y'].values


# In[ ]:


#Splitting the Train and Test Data


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[7]:


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1], 'C': [1]},
                    {'kernel': ['linear'], 'C': [1]}]


# In[8]:


clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision')
clf.fit(X_train, y_train)


# In[10]:


print('Model Selected is: ', clf.best_params_ ,'Which Produces a precision of ', clf.best_score_)


# In[ ]:





# In[11]:


from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
y_true, y_pred = y_test, clf.predict(X_test)
pre1 = precision_score(y_true, y_pred)
rec1 = recall_score(y_true, y_pred)
acc1 = accuracy_score(y_true, y_pred)
f1_1 = f1_score(y_true, y_pred)
print('precision on the evaluation set: ', pre1)
print('recall on the evaluation set: ', rec1)
print('accuracy on the evaluation set: ', acc1)


# In[13]:


from sklearn.decomposition import PCA
# raw data
X = da.drop('y', axis=1).values
y = da['y'].values
# split, random_state is used for repeatable results, you should remove it if you are running your own code.
pca = PCA(n_components=0.9)
x_pca = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.30, random_state=42)
x_pca.shape


# In[14]:


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1],
                     'C': [1]},
                    {'kernel': ['linear'], 'C': [1]}]


# In[15]:


clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision')
clf.fit(X_train, y_train)


# In[16]:


print('Model Selected is: ', clf.best_params_ ,'Which Produces a precision of ', clf.best_score_)


# In[17]:


y_true, y_pred = y_test, clf.predict(X_test)
pre2 = precision_score(y_true, y_pred)
rec2 = recall_score(y_true, y_pred)
acc2 = accuracy_score(y_true, y_pred)
f1_2 = f1_score(y_true, y_pred)
print('precision on the evaluation set: ', pre2)
print('recall on the evaluation set: ', rec2)
print('accuracy on the evaluation set: ', acc2)


# In[18]:


table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score Without PCA': [acc1, pre1, rec1, f1_1],
    'Score With PCA': [acc2, pre2, rec2, f1_2]
    })
table


# In[ ]:




