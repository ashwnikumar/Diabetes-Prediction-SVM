#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[9]:


df=pd.read_csv('C:/Users/KUSH/jupyter/diabetes.csv')


# In[10]:


#Prints first 5 rows 
df.head()


# In[13]:


#print No of Rows and Columns in the Dataset
df.shape


# In[12]:


#Get Statistical measures of dataset
df.describe()


# In[15]:


#Number of 0's and 1's in outcome column
df['Outcome'].value_counts()


# In[17]:


# Mean Value of various column filtered by 'Outcome'
df.groupby('Outcome').mean()


# In[19]:


# Separating data and labels 
# Set axis =1 when dropping a column
# Set axis =0 when dropping a row
X=df.drop(columns='Outcome',axis=1)
Y=df['Outcome']


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


# Standardizing Dataset to bring them in same Range 
scaler=StandardScaler()


# In[23]:


scaler.fit(X)


# In[26]:


standardized_data=scaler.transform(X)


# In[27]:


X=standardized_data
Y=df['Outcome']


# In[28]:


print(X)


# In[29]:


# Split Train /Test Data
# Stratify=Y To ensure even distribution of diabetec cases in both Test and train datasets
# random_state for replicating code ,if set to 1 the datset may be separated in another way
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[30]:


print(X.shape,X_train.shape,X_test.shape)


# In[33]:


#Training the Model
classifier=svm.SVC(kernel='linear')


# In[34]:


#Training the SVM classifier 
classifier.fit(X_train,Y_train)


# In[35]:


#Model Evaluation
X_train_prediction=classifier.predict(X_train)


# In[37]:


training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[38]:


print('Accuracy Score of training data',training_data_accuracy)


# In[39]:


# Accuracy Score On Test Data 
X_test_prediction=classifier.predict(X_test)


# In[40]:


test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[41]:


print('Accuracy Score of training data',test_data_accuracy)


# In[45]:


#Predictive System
input_data=(5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array=np.asarray(input_data)
#Reshaping the data 
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


# In[48]:


#standarize the I/P data
std_data=scaler.transform(input_data_reshaped)
prediction =classifier.predict(std_data)
if (prediction[0]==1):
    print("The person has",int(test_data_accuracy*100)," % chances of having diabetes")
else:
    print("The person is Non Diabetic")


# In[ ]:




