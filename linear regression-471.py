#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter("ignore")


# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[46]:


data=pd.read_csv("Admission.csv")
data


# In[47]:


data=data.loc[::,['GPA','GMAT']]
data


# In[48]:


x=data.iloc[:,0]


# In[49]:


x.shape


# In[50]:


x=data.iloc[:,0].values.reshape(-1,1)


# In[51]:


x.shape


# In[52]:


y=data.iloc[:,-1].values.reshape(-1,1)


# In[53]:


y.shape


# In[54]:


x


# In[55]:


y


# In[56]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


plt.scatter(x,y)
plt.show


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[62]:


x_test.shape


# In[63]:


y_train.shape


# In[64]:


x_test.shape


# In[65]:


x_train.shape


# In[66]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)


# In[67]:


y_pred = lm.predict(x_test)
y_pred


# In[68]:


plt.scatter(x,y,color = 'blue')
plt.plot(x_test,y_pred,color = 'red')


# In[ ]:




