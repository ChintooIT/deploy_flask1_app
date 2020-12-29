#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
#from statistics import mean 

#import seaborn as sns
import matplotlib.pyplot as plt

#from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV


# In[2]:


df=pd.read_csv("./loan_train_data.csv")
df.head()


# In[3]:


df.drop(['Unnamed: 0'],axis=1, inplace=True)


# In[4]:


df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])


# In[5]:


df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mode()[0])


# In[6]:


df['Married']=df['Married'].fillna(df['Married'].mode()[0])


# In[7]:


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])


# In[8]:


df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[9]:


df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])


# In[10]:


df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mode()[0])


# In[11]:


df.nunique()


# In[12]:


df.info()


# In[13]:


df.drop(['Loan_ID'], axis=1, inplace=True)


# In[14]:


df['Gender']=pd.factorize(df['Gender'])[0]


# In[15]:


df['Married']=pd.factorize(df['Married'])[0]


# In[16]:


df['Dependents']=pd.factorize(df['Dependents'])[0]


# In[17]:


df['Education']=pd.factorize(df['Education'])[0]


# In[18]:


df['Self_Employed']=pd.factorize(df['Self_Employed'])[0]


# In[19]:


df['Property_Area']=pd.factorize(df['Property_Area'])[0]


# In[20]:


df.head()


# In[21]:


df.shape


# In[22]:


X_train= df.drop('Loan_Status', axis = 1)              # Input Variables/features
y_train= df.Loan_Status 


# In[23]:


# import SMOTE 
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 25, sampling_strategy = 1.0)  


# In[24]:


# fit the sampling
X_train, y_train = sm.fit_sample(X_train, y_train)


# In[25]:


from sklearn.ensemble import RandomForestClassifier


rfc= RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[26]:


df1=pd.read_csv("./loan_test_data.csv")
df1.head()


# In[27]:


df1.drop(['Loan_ID'], axis=1, inplace=True)


# In[28]:


df1.isnull().sum()


# In[29]:


df1['Dependents']=df1['Dependents'].fillna(df1['Dependents'].mode()[0])


# In[30]:


df1['LoanAmount']=df1['LoanAmount'].fillna(df1['LoanAmount'].mode()[0])


# In[31]:


df1['Married']=df1['Married'].fillna(df1['Married'].mode()[0])


# In[32]:


df1['Gender']=df1['Gender'].fillna(df1['Gender'].mode()[0])


# In[33]:


df1['Self_Employed']=df1['Self_Employed'].fillna(df1['Self_Employed'].mode()[0])


# In[34]:


df1['Loan_Amount_Term']=df1['Loan_Amount_Term'].fillna(df1['Loan_Amount_Term'].mode()[0])


# In[35]:


df1['Credit_History']=df1['Credit_History'].fillna(df1['Credit_History'].mode()[0])


# In[36]:


df1.nunique()


# In[37]:


df1['Gender']=pd.factorize(df1['Gender'])[0]


# In[38]:


df1['Married']=pd.factorize(df1['Married'])[0]


# In[39]:


df1['Dependents']=pd.factorize(df1['Dependents'])[0]


# In[40]:


df1['Education']=pd.factorize(df1['Education'])[0]


# In[41]:


df1['Self_Employed']=pd.factorize(df1['Self_Employed'])[0]


# In[42]:


df1['Property_Area']=pd.factorize(df1['Property_Area'])[0]


# In[43]:


df1.shape


# In[44]:


X_test =df1[:]


# In[46]:


y_pred = rfc.predict(X_test)


# In[ ]:


import pickle
filename='loan_model.pkl'
pickle.dump(rfc,open(filename,'wb'))


# In[ ]:


#import pickle
##saving the model
#with open("model.bin", 'wb') as f_out:
#     pickle.dump(final_model, f_out)
#     f_out.close()


# In[ ]:


##loading the model from the saved file
# with open('model.bin', 'rb') as f_in:
#     model = pickle.load(f_in)

# predict_mpg(vehicle_config, model)

