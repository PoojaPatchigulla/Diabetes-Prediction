#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split


# In[2]:


raw_data = pd.read_csv('diabetes2.csv')
raw_data.head()


# In[3]:


raw_data.describe()


# In[4]:


for i in raw_data.columns:
    raw_data[i].hist(bins=40, figsize=(5,5))
    plt.suptitle(i)
    plt.show()


# In[5]:


#we dont have null values
raw_data.info()


# In[6]:


sns.heatmap(raw_data.corr(numeric_only=True), cmap='YlGnBu')
plt.show()


# In[7]:


processed_data = raw_data.copy()
processed_data.head()


# In[8]:


X = processed_data.iloc[:,0:8]
y = processed_data.iloc[:,8:9]


# In[9]:


class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        self.scaler = StandardScaler()
        return self
    def transform(self, X):
        X = X.copy()
        for feature in X.columns:
            X[feature] = self.scaler.fit_transform(X[[feature]])
        return X


# In[10]:


pipeline = Pipeline(
    [('featurencoder',FeatureEncoder())]
)


# In[11]:


X = pipeline.fit_transform(X)
X.head()


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[13]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Model Score',  model.score(X_train,y_train))
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[15]:


model.intercept_


# In[16]:


model.coef_


# In[17]:


# Create summary table with coeef and feat
feature_name = X.columns.values
summary_table = pd.DataFrame(columns=['Feature name'], data = feature_name)
summary_table['Coefficient'] = np.transpose(model.coef_)
summary_table


# In[18]:


#Add Intercept
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', model.intercept_[0]]


# In[19]:


summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)


# In[20]:


summary_table.sort_values('Odds_ratio', ascending=False)


# In[ ]:



