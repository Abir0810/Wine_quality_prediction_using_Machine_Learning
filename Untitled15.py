#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[5]:


df=pd.read_csv(r"E:\ai\JNU Masters\Wine Quality\winequality-white.csv")


# In[6]:


df.head(5)


# In[ ]:





# In[7]:


data={
    
    'fixed acidity' : [8.03,7.63,8.33,8.13],
    'volatile acidity' : [1.31,1.26,1.32,1.29],
    'citric acid': [0.43,0.44,0.44,0.45],
    'residual sugar' : [1.9,2.2,1.7,2.3],
    'chlorides' : [0.52,0.62,0.51,0.5],
    'free sulfur dioxide' : [25.43,15.43,22.43,11.43],
    'total sulfur dioxide' : [67.43,60.43,57.43,38.43],
    'density' :  [1.421,1.426,1.427,1.424],
    'pH': [3.22,3.52,3.26,3.12],
    'sulphates' : [1.11,0.98,1.07,0.51],
    'alcohol' : [10.23,10.03,10.23,9.83],
    'quality' : [5,6,2,3]
}


# In[8]:


new_data = df.append(data, ignore_index=True)


# In[9]:


df.head(5)


# In[10]:


df = pd.DataFrame(data)


# In[11]:


df.to_csv('GFG.csv', mode='a', index=False, header=False)


# In[12]:


df.head(3)


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.isnull().sum()


# In[16]:


sb.pairplot(df)
plt.show()


# In[17]:


df.hist(bins=20,figsize=(10,10))
plt.show()


# In[18]:


plt.figure(figsize=[15,6])
plt.bar(df['quality'],df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[19]:


plt.figure(figsize=[18,7])
sb.heatmap(df.corr(),annot=True)
plt.show()


# In[24]:


new_df = df.drop('total sulfur dioxide',axis = 1)


# In[25]:


new_df.update(new_df.fillna(new_df.mean()))


# In[26]:


cat = new_df.select_dtypes(include='O')
df_dummies = pd.get_dummies(new_df,drop_first = True)
print(df_dummies)


# In[28]:


df_dummies['best quality']=[1 if x>=7 else 0 for x in df.quality]
print(df_dummies)


# In[ ]:





# In[31]:


from sklearn.model_selection import train_test_split


# In[33]:


x = df_dummies.drop(['quality','best quality'],axis=1)
y = df_dummies['best quality']


# In[35]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=40)


# In[36]:


# code
# import libraries

from sklearn.preprocessing import MinMaxScaler
# creating scaler scale var.
norm = MinMaxScaler()
# fit the scale
norm_fit = norm.fit(xtrain)
# transformation of training data
scal_xtrain = norm_fit.transform(xtrain)
# transformation of testing data
scal_xtest = norm_fit.transform(xtest)
print(scal_xtrain)


# In[47]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report


# In[48]:


rnd = RandomForestClassifier()
fit_rnd = rnd.fit(xtrain,ytrain)
rnd_score = rnd.score(xtest,ytest)


# In[49]:


print('score of model is : ',rnd_score)
print('.................................')
print('calculating the error')


# In[54]:


x_predict = list(rnd.predict(xtest))
df = {'predicted':x_predict,'original':ytest}
pd.DataFrame(df).head(10)


# In[ ]:




