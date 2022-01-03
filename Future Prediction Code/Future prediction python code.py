#!/usr/bin/env python
# coding: utf-8

# #                               FUTURE PREDICTION MODEL

# Firstly, we are importing pandas and numpy module.Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool. Numpy refers to numeric python.

# In[1]:


import pandas as pd
import numpy as np


# # Reading the dataset

# We are reading the csv file and storing it as dataframe 

# In[2]:


df=pd.read_csv("Dataset - Sheet3.csv")


# # Data pre-processing

# head() method - Returns the first 5 rows of the dataframe.

# In[3]:


df.head()


# tail() method -  Returns the last 5 rows of the dataframe.

# In[4]:


df.tail()


# Columns attribute return the column labels of the given Dataframe.

# In[5]:


df.columns


# dtypes attributes returns a Series with the data type of each column.

# In[6]:


df.dtypes


# describe() method is used to view some basic statistical details like percentile, mean, std etc. of a data frame

# In[7]:


df.describe()


# info() function is used to print a concise summary of a DataFrame. This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.

# In[8]:


df.info()


# Using the below snippet code we are converting the datatypes of the columns(object-->int).

# In[9]:


from sklearn.preprocessing import LabelEncoder
category= ['Date','Day','Time'] 
encoder= LabelEncoder()
for i in category:   
    df[i] = encoder.fit_transform(df[i]) 
df.dtypes


# corr() is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored. Note: The correlation of a variable with itself is 1.

# In[10]:


df.corr()


# From the above result we can conclude that all variables are independent of each other. So the idea of linear regression also fails 

# *values attribute returns the numpy representation of the given DataFrame.
# 'Date','Day','Time' are taken into X and these independent variables 

# In[11]:


X=df[['Date','Day','Time']].values
X[0:5]    #we are converting dataframe into numpy arrays.


# *values attribute returns the numpy representation of the given DataFrame.
# 'Empty level in cm(Total size=100cm)' are taken into Y and it is a dependent variables 

# In[12]:


Y=df['Empty level in cm(Total size=100cm)'].values      #we are converting dataframe into numpy arrays.
Y[0:5]


# Using the below snippet code we are converting the datatype of the X(int64-->float).

# In[13]:


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# # Data Splitting

# From the below snippet code we are dividing our dataset into two parts and that are 1.train set 2.test set basically test set will be 20% of the dataset
# Assume a dataset contains 100 data points then 80 data points are used as training set and 20 data points as test set. 

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # Model Training

# We are using DecisionTreeClassifier for training the model.
# Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
# DecisionTreeClassifier takes as input two arrays: an array X, sparse or dense, of shape (n_samples, n_features) holding the training samples, and an array Y of integer values, shape (n_samples,), holding the class labels for the training samples

# In[15]:


from sklearn.tree import DecisionTreeClassifier


# We are creating a object of DecisionTreeClassifier named model

# In[16]:


model = DecisionTreeClassifier()

# fit the model with the training data
model.fit(X_train,y_train)


# We are using the trained model and predicting the test set.

# In[17]:


y_pred = model.predict(X_test)


# Using the below snippet code we are calculating r2_score, mse, and accuracy of our model.

# In[18]:


import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
accuracy = metrics.accuracy_score(y_pred,y_test)
print("Accuracy : %s" % "{0:.3%}".format(accuracy))            #Accuracy of the model
print("r2_score:",r2_score(y_test, y_pred))                      #R2 value
print("mse:",mean_squared_error(y_test, y_pred))                 #MSE(Mean Square Error) of the model


# Trained model is almost 80% accurate.

# # Predicting the future values based on trained model. 

# In[20]:


predict=pd.read_csv("prediction.csv")


# In[21]:


from sklearn.preprocessing import LabelEncoder
category= ['Date','Day','Time'] 
encoder= LabelEncoder()
for i in category:   
    predict[i] = encoder.fit_transform(predict[i]) 
predict.dtypes


# In[22]:


N=predict[['Date','Day','Time']].values
N[0:5]


# In[23]:


from sklearn import preprocessing
N = preprocessing.StandardScaler().fit(N).transform(N.astype(float))
N[0:5]


# In[28]:


yhat = model.predict(N)


# In[29]:


print(yhat)


# In[32]:


predict["Empty level in cm(Total size=100cm)"]=yhat


# In[27]:


predict.head()


# In[31]:


predict["Day"].replace({1:"Monday",5:"Tuesday",6:"Wednesday",4:"Thursday",0:"Friday",2:"Saturday",3:"Sunday"},inplace=True)


# In[33]:


predict.to_csv("Prediction_output.csv")

