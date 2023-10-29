#!/usr/bin/env python
# coding: utf-8

# This script contains the following:
# 1. Importing your libraries and data
# 2. Subsetting, wrangling, and cleaning time-series data
# 3. Time series analysis: decomposition
# 4. Testing for stationarity
# 5. Stationarizing the U.S. GDP data

# In[1]:


#1. Importing your libraries and data

import quandl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import os
import warnings
import nasdaqdatalink

warnings.filterwarnings("ignore") # Disable deprecation warnings that could indicate, for instance, a suspended library or 
# feature. These are more relevant to developers and very seldom to analysts.

plt.style.use('fivethirtyeight') 


# In[2]:


# Importing the US GDP from FRED,as the project working on has U.S. geographic involved.

data = nasdaqdatalink.get("FRED/GDP")


# In[3]:


data.head(5)


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


type(data)


# In[7]:


# Plot the data using matplotlib.

plt.figure(figsize=(15,5), dpi=100) 
# The dpi argument controls the quality of the visualization here. When it's set to 100,
# it will produce lower-than-standard quality, which is useful if, similar to this notebook, will have a lot of plots.
# A large number of plots will increase the size of the notebook, which could take more time to load and eat up a lot of RAM!

plt.plot(data)


# In[8]:


#2. Subsetting, wrangling, and cleaning time series data.

# Reset index so that you can use the "Date" column as a filter

data_2 = data.reset_index()


# In[9]:


data_2.head()


# In[10]:


#Subsetting the data from 1980 to 2020.
data_sub = data_2.loc[(data_2['Date'] >= '1980-01-01') & (data_2['Date'] < '2020-06-01')]


# In[11]:


data_sub.shape


# In[12]:


data_sub.head()


# In[13]:


# Set the "Date" column as the index

from datetime import datetime

data_sub['datetime'] = pd.to_datetime(data_sub['Date']) # Create a datetime column from "Date.""
data_sub = data_sub.set_index('datetime') # Set the datetime as the index of the dataframe.
data_sub.drop(['Date'], axis=1, inplace=True) # Drop the "Date" column.
data_sub.head()


# In[14]:


# Plot the new data set

plt.figure(figsize=(15,5), dpi=100)
plt.plot(data_sub)


# In[15]:


# Check for missing values, no missing values

data_sub.isnull().sum() 


# In[16]:


# Check for duplicates

dups = data_sub.duplicated()
dups.sum()

# No duplicates


# In[17]:


#3. Time-series analysis: decomposition

# Decompose the time series using an additive model

decomposition = sm.tsa.seasonal_decompose(data_sub, model='additive')


# In[18]:


from pylab import rcParams # This will define a fixed size for all special charts.

rcParams['figure.figsize'] = 18, 7


# In[19]:


# Plot the separate components

decomposition.plot()
plt.show()


# There is a upward linear trend in the data, which appears similar to the level as this data did not require smoothing. We can also see from the decomposition that there is seasonality present in this data, represented by the spiked curve that changes at regular intervals. Finally, the residual chart shows the noise of the data, which shows the plots are fairly closely centered around zero, there isn't an immense amount. 

# In[20]:


#4. Testing for stationarity(The Dickery_Fuller test)

# The adfuller() function will import from the model from statsmodels for the test; however, running it will only return 
# an array of numbers. This is why you need to also define a function that prints the correct output from that array.

from statsmodels.tsa.stattools import adfuller # Import the adfuller() function

def dickey_fuller(timeseries): # Define the function
    # Perform the Dickey-Fuller test:
    print ('Dickey-Fuller Stationarity test:')
    test = adfuller(timeseries, autolag='AIC')
    result = pd.Series(test[0:4], index=['Test Statistic','p-value','Number of Lags Used','Number of Observations Used'])
    for key,value in test[4].items():
       result['Critical Value (%s)'%key] = value
    print (result)

# Apply the test using the function on the time series
dickey_fuller(data_sub['Value'])


# In[21]:


# Check out a plot of autocorrelations

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Here, you import the autocorrelation and partial correlation plots

plot_acf(data_sub)
plt.show()


# There are many lags above the confidence interval edge, which means there are many lags significantly correlated with each other (or there is a lot of autocorrelated data, supporting the result of the Dickey-Fuller test)

# In[22]:


#5. Stationarizing the US GDP data.

data_diff = data_sub - data_sub.shift(1) 
# The df.shift(1) function turns the observation to t-1, making the whole thing t - (t -1)


# In[23]:


data_diff.dropna(inplace = True) 
# Here, you remove the missing values that came about as a result of the differencing. 
# You need to remove these or you won't be able to run the Dickey-Fuller test.


# In[24]:


data_diff.head()


# In[25]:


data_diff.columns


# In[26]:


# Check out what the differencing did to the time-series curve

plt.figure(figsize=(15,5), dpi=100)
plt.plot(data_diff)


# In[27]:


dickey_fuller(data_diff)


# In[28]:


#2nd round (Additional round of differencing)

data_diff = data_sub - data_sub.shift(2) 
# The df.shift(1) function turns the observation to t-1, making the whole thing t - (t -1)


# In[29]:


data_diff.dropna(inplace = True) 
# Here, you remove the missing values that came about as a result of the differencing. 
# You need to remove these or you won't be able to run the Dickey-Fuller test.


# In[30]:


data_diff.head()


# In[31]:


data_diff.columns


# In[32]:


# Check out what the differencing did to the time-series curve

plt.figure(figsize=(15,5), dpi=100)
plt.plot(data_diff)


# In[33]:


dickey_fuller(data_diff)


# In[34]:


#3nd round (Additional round of differencing)

data_diff = data_sub - data_sub.shift(3) 
# The df.shift(1) function turns the observation to t-1, making the whole thing t - (t -1)


# In[35]:


data_diff.dropna(inplace = True) 
# Here, you remove the missing values that came about as a result of the differencing. 
# You need to remove these or you won't be able to run the Dickey-Fuller test.


# In[36]:


data_diff.head()


# In[37]:


data_diff.columns


# In[38]:


# Check out what the differencing did to the time-series curve

plt.figure(figsize=(15,5), dpi=100)
plt.plot(data_diff)


# In[39]:


dickey_fuller(data_diff)


# Test statistic is now smaller than both the 5% and 10% Critical Values, after two additional rounds of differencing.

# In[40]:


plot_acf(data_diff)
plt.show()


# While there are still a few lags above the blue confidence interval, there are less than 10 of them so I will choose to stop here!

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




