#!/usr/bin/env python
# coding: utf-8

# This script contains the following:
# 1. Importing libraries and data and renaming columns
# 2. The elbow technique
# 3. k-means clustering
# 

# In[25]:


#Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.cluster import KMeans # Here is where you import the k-means algorithm from scikit-learn.
import pylab as pl # PyLab is a convenience module that bulk imports matplotlib.


# In[26]:


# This option ensures the graphs you create are displayed in your notebook without the need to "call" them specifically.

get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


path =r'/Users/angitha/Achievement 6.5/'


# In[28]:


df = pd.read_csv(os.path.join(path, '02.Data', 'Original data', 'cases_deaths_gender_cleaned.csv'))


# In[29]:


df.shape


# In[30]:


df.head()


# In[31]:


# Create a subset including required variables
df = df[['C.cases','C.deaths','Pop.male','Pop.female','median_age','T.population','female_percentage']]


# In[32]:


#2. The elbow technique
num_cl = range(1, 10) # Defines the range of potential clusters in the data.
kmeans = [KMeans(n_clusters=i,n_init='auto') for i in num_cl] # Defines k-means clusters in the range assigned above.


# In[33]:


score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))] 
# Creates a score that represents 
# a rate of variation for the given cluster option.

score


# In[34]:


# Plot the elbow curve using PyLab.

pl.plot(num_cl,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# There's a large jump from two to three and three to four on the x-axis, but after that, the curve straightens out. This means that the optimal count for your clusters is FOUR.

# In[35]:


#3. k-means clustering
# Create the k-means object.

kmeans = KMeans(n_clusters = 4,n_init='auto') 


# In[36]:


# Fit the k-means object to the data.

kmeans.fit(df)


# In[37]:


df['clusters'] = kmeans.fit_predict(df)


# In[38]:


df.head()


# In[39]:


df['clusters'].value_counts()


# In[40]:


# Plot the clusters for the "median_age" and "C.cases" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['median_age'], y=df['C.cases'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.

ax.grid(False) # This removes the grid from the background.
plt.xlabel('Median age') # Label x-axis.
plt.ylabel('COVID cases') # Label y-axis.
plt.show()


# In[41]:


# Plot the clusters for the "median_age" and "C.deaths" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['median_age'], y=df['C.deaths'], hue=kmeans.labels_, s=100)

ax.grid(False) 
plt.xlabel('Median age') 
plt.ylabel('COVID cases') 
plt.show()


# In[42]:


# Plot the clusters for the "median_age" and "T.population" variables.

plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df['median_age'], y=df['T.population'], hue=kmeans.labels_, s=100)

ax.grid(False) 
plt.xlabel('Median age') 
plt.ylabel('Total population') 
plt.show()


# 8.Discuss how and why the clusters make sense.
#     The clusters clearly inclines to the hypothesis that the population at the age group of 30 to 50 are suceptible to COVID virus infections. 

# In[45]:


df.loc[df['clusters'] == 3, 'cluster'] = 'brown'
df.loc[df['clusters'] == 2, 'cluster'] = 'dark purple'
df.loc[df['clusters'] == 1, 'cluster'] = 'purple'
df.loc[df['clusters'] == 0, 'cluster'] = 'pink'


# In[46]:


df.groupby('cluster').agg({'median_age':['mean', 'median'], 
                         'C.cases':['mean', 'median'], 
                         'C.deaths':['mean', 'median'],
                          'T.population':['mean', 'median']})


# 10.Propose what these results could be useful for in future steps of an analytics pipeline.
#     The results could be used for predictive analysis in future. Cluster analysis is helpful in this dataset as the data has non-linear relationships. 
# 

# In[ ]:





# In[ ]:




