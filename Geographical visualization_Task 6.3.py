#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Visualization Libraries and Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import folium
import json


# In[2]:


# This command propts matplotlib visuals to appear in the notebook 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Import ".json" file for the U.S
country_geo=r'/Users/angitha/Downloads/us-states.json'


# In[4]:


country_geo


# In[5]:


# Define path
path = r'/Users/angitha/Achievement 6.3/'


# In[6]:


# Import data
df = pd.read_csv(os.path.join(path, '02.Data', 'Original data', 'cases_deaths_gender.csv'))


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


#Rename columns whose names are not appropriate

df.rename(columns = {'cases' : 'C.cases', 'deaths': 'C.deaths', 
                     'population': 'T.population', 'male': 'Pop.male',
                     'female': 'Pop.female'},
                      inplace = True)


# In[10]:


df.dtypes


# In[11]:


df.head()


# In[12]:


# Check for missing values
df.isnull().sum() # 85399 missing values in column state_code!


# In[13]:


#drop columns as it is only approximately 2% of the data
df=df.dropna()


# In[14]:


# Check for missing values
df.isnull().sum() 
# Dropped rows with missing values in column state_code!


# In[15]:


# Find duplicates
df_dups = df[df.duplicated()]


# In[16]:


df_dups.shape # No duplicates


# In[17]:


#No mixed-type columns
for col in df.columns.tolist():
      weird = (df[[col]].applymap(type) != df[[col]].iloc[0].apply(type)).any(axis = 1)
      if len (df[weird]) > 0:
        print (col)


# In[18]:


# Create a subset including required variables

df = df[['state','C.cases','C.deaths','Pop.male','Pop.female','median_age','T.population','female_percentage']]


# In[19]:


#Extreme value check
sns.histplot(df['C.cases'], bins=20, kde = True) # doesnt show extreme values for 'C.cases'


# In[20]:


#Extreme value check
sns.histplot(df['C.deaths'], bins=20, kde = True) # doesnt show extreme values for 'C.deaths'


# In[21]:


#Extreme value check
sns.histplot(df['T.population'], bins=20, kde = True) # doesnt show extreme values for 'T.population'


# In[36]:


# Create a data frame with just the states and the values for we want plotted

data_to_plot = df[['state','C.cases']]
data_to_plot.head()


# In[37]:


# Setup a folium map at a high-level zoom
map = folium.Map(location = [100, 0], zoom_start = 1.5)

# Choropleth maps bind Pandas Data Frames and json geometries.This allows us to quickly visualize data combinations
folium.Choropleth(
    geo_data = country_geo, 
    data = data_to_plot,
    columns = ['state', 'C.cases'],
    key_on = 'feature.properties.name', # this part is very important - check your json file to see where the KEY is located
    fill_color = 'YlOrBr', fill_opacity=0.6, line_opacity=0.1,
    legend_name = "rating").add_to(map)
folium.LayerControl().add_to(map)

map


# In[38]:


map.save('plot_data.html')


# In[39]:


# Create a data frame with just the states and the values for we want plotted

data_to_plot = df[['state','C.deaths']]
data_to_plot.head()


# In[40]:


# Setup a folium map at a high-level zoom
map = folium.Map(location = [100, 0], zoom_start = 1.5)

# Choropleth maps bind Pandas Data Frames and json geometries.This allows us to quickly visualize data combinations
folium.Choropleth(
    geo_data = country_geo, 
    data = data_to_plot,
    columns = ['state', 'C.deaths'],
    key_on = 'feature.properties.name', # this part is very important - check your json file to see where the KEY is located
    fill_color = 'YlOrBr', fill_opacity=0.6, line_opacity=0.1,
    legend_name = "rating").add_to(map)
folium.LayerControl().add_to(map)

map


# In[41]:


map.save('plot_data1.html')


# In[42]:


# Create a data frame with just the states and the values for we want plotted

data_to_plot = df[['state','T.population']]
data_to_plot.head()


# In[44]:


# Setup a folium map at a high-level zoom
map = folium.Map(location = [100, 0], zoom_start = 1.5)

# Choropleth maps bind Pandas Data Frames and json geometries.This allows us to quickly visualize data combinations
folium.Choropleth(
    geo_data = country_geo, 
    data = data_to_plot,
    columns = ['state', 'T.population'],
    key_on = 'feature.properties.name', # this part is very important - check your json file to see where the KEY is located
    fill_color = 'YlOrBr', fill_opacity=0.6, line_opacity=0.1,
    legend_name = "rating").add_to(map)
folium.LayerControl().add_to(map)

map


# In[45]:


map.save('plot_data2.html')


# In[46]:


# Create a data frame with just the states and the values for we want plotted

data_to_plot = df[['state','median_age']]
data_to_plot.head()


# In[47]:


# Setup a folium map at a high-level zoom
map = folium.Map(location = [100, 0], zoom_start = 1.5)

# Choropleth maps bind Pandas Data Frames and json geometries.This allows us to quickly visualize data combinations
folium.Choropleth(
    geo_data = country_geo, 
    data = data_to_plot,
    columns = ['state', 'median_age'],
    key_on = 'feature.properties.name', # this part is very important - check your json file to see where the KEY is located
    fill_color = 'YlOrBr', fill_opacity=0.6, line_opacity=0.1,
    legend_name = "rating").add_to(map)
folium.LayerControl().add_to(map)

map


# In[48]:


map.save('plot_data3.html')


# ANSWERS
# Does the analysis answer any of your existing research questions?
#     Yes, it does. The three choropleth maps clearlt shows the answer to the question whether the population size 
# affect the percentage of COVID cases and Covid related deaths. The states with hihest population are Boston, 
# Pennsylvania, Phoenix, Kansas, Montana, Utah, South Carolina, Seattle and Maine. All these states have higher 
# cases and deaths due to COVID. 
# It also shows that the states with higher COVID death and cases have more population below the age of 40 years. 
# 
# Does the analysis lead you to any new research questions?
#  The data regarding hospitals in each state could help in analysing the need for more medical facilities. 
# 

# In[ ]:




