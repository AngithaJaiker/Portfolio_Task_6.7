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


# In[2]:


matplotlib.__version__


# In[3]:


# This option ensures the charts you create are displayed in the notebook without the need to "call" them specifically.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Define path
path = r'/Users/angitha/Achievement 6.2/'


# In[5]:


# Import data
df = pd.read_csv(os.path.join(path, '02.Data', 'Original data', 'cases_deaths_gender.csv'))


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


#Rename columns whose names are too long

df.rename(columns = {'cases' : 'C.cases', 'deaths': 'C.deaths', 
                     'population': 'T.population', 'male': 'Pop.male',
                     'female': 'Pop.female'},
                      inplace = True)


# In[9]:


df.dtypes


# In[10]:


df.groupby('county').agg({'county_code': ['mean']})


# In[11]:


df.head()


# In[12]:


# Check for missing values
df.isnull().sum() # 85399 missing values in column state_code!


# In[13]:


# Find duplicates
df_dups = df[df.duplicated()]


# In[14]:


df_dups.shape # No duplicates


# In[15]:


for col in df.columns.tolist():
      weird = (df[[col]].applymap(type) != df[[col]].iloc[0].apply(type)).any(axis = 1)
      if len (df[weird]) > 0:
        print (col)
        
# state_code is a mixed-type column.


# In[16]:


#drop columns
df=df.dropna()


# In[17]:


# Check for missing values
df.isnull().sum() 
# Dropped rows with missing values in column state_code!


# In[18]:


# Create a subset excluding the Unnamed:0, county,state,date,state_code,date and region columns

df = df[['county_code','lat','long','C.cases','C.deaths','Pop.male','Pop.female','median_age','T.population','female_percentage']]


# In[19]:


# Create a correlation matrix using pandas

df.corr()


# In[20]:


# Create a correlation heatmap using matplotlib

plt.matshow(df.corr())
plt.show()


# In[21]:


# Create a correlation heatmap using matplotlib

plt.matshow(df.corr())
plt.show()


# In[22]:


# Save figure
plt.matshow(df.corr())
plt.savefig("out.png") 

# This will save the image in the working directory. 
#If you don't know what this directory is the next line will show you how to check


# In[23]:


#current dir
cwd = os.getcwd()
cwd


# In[24]:


# Add labels, a legend, and change the size of the heatmap

f = plt.figure(figsize=(8, 8)) # figure size 
plt.matshow(df.corr(), fignum=f.number) # type of plot
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45) # x axis labels
plt.yticks(range(df.shape[1]), df.columns, fontsize=14) # y axis labels
cb = plt.colorbar() # add a colour legend (called colorbar)
cb.ax.tick_params(labelsize=12) # add font size
plt.title('Correlation Matrix', fontsize=14) # add title


# In[25]:


df.columns


# In[26]:


# Create a subset excluding the "Date" and "Unnamed:0" columns

sub = df[['county_code','lat','long','C.cases','C.deaths','Pop.male','Pop.female','median_age','T.population','female_percentage']]


# In[27]:


sub


# In[28]:


# Create a subplot with matplotlib
f,ax = plt.subplots(figsize=(10,10))

# Create the correlation heatmap in seaborn by applying a heatmap onto the correlation matrix and the subplots defined above.
corr = sns.heatmap(sub.corr(), annot = True, ax = ax) # The `annot` argument allows the plot to 
#place the correlation coefficients onto the heatmap.


# In[29]:


df_small=df[0:1000000]


# In[30]:


df_small


# In[31]:


# Create a scatterplot for the "median_age" and "C.deaths" columns in seaborn

sns.lmplot(x = 'median_age', y = 'C.deaths', data = df_small)


# In[32]:


# Keep only the variables you want to use in the pair plot

df_small_2 = df_small[['county_code', 'C.deaths', 'C.cases', 'T.population']]


# In[33]:


# Create a pair plot 

g = sns.pairplot(df_small_2)


# In[34]:


# Use a histogram to visualize the distribution of the variables. 
# This way, you can determine sensible categories for the price ranges. 
# You don't want to end up with too few observations in any one of the categories.
# The argument "kde" add a line that encompasses the distribution

sns.histplot(df['median_age'], bins = 20, kde = True)


# In[41]:


df.loc[df['median_age'] < 40, 'Risk_category'] = 'High risk'


# In[42]:


df.loc[(df['median_age'] >= 40) & (df['median_age'] < 60), 'Risk_category'] = 'Moderate risk'


# In[43]:


df.loc[df['median_age'] >= 60, 'Risk_category'] = 'Low risk'


# In[45]:


df['Risk_category'].value_counts(dropna = False)


# In[46]:


df.columns


# In[49]:


# Create a categorical plot in seaborn using the Risk_category created above

sns.set(style="ticks")
g = sns.catplot(x="C.cases", y="median_age", hue="Risk_category", data=df)


# In[50]:


# fig = g.get_figure()
g.savefig("out.png") 

# Again, the image will be saved in the working directory. 


# In[52]:


# Create a categorical plot in seaborn using the Risk_category created above

sns.set(style="ticks")
g = sns.catplot(x="C.deaths", y="median_age", hue="Risk_category", data=df)


# # ANSWERS(A reduced dataframe size has been used due to technical difficulty)

# HEATMAP(Colored)
# 3.Discuss what the coefficients in the plot mean in terms of the relationships between the variables?
# The coefficients in the plot shows the correlation between Covide related deaths and cases in the U.S.
# in different counties.
# It shows that the death and cases of COVID are negatively correlated to the median age of the population. 
# It means that more than the elderly population, the younger population is more impacted with the novel virus.
# The number of deaths and cases are positively correlated. 
# Counties with higher population has higher number of COVID cases and deaths too. 
# Also there seems to be a slight positive correlation for deaths and COVID cases to percentage of female population.
# 

# SCATTERPLOT
# 4.Discuss the output in a markdown cell?
# Scatterplot was drawn against the variables median age and COVID deaths. 
# Its tightly bound and indicates an inverse proportion, which tells that the number of COVID
# related deaths were fewer among elderly people, population above 60years of age. 

# PAIR PLOT
# 5.Comment on the distribution of the variables and mark variables you’d like to explore further with an explanation of why?
# The pair plot shows that the higher the population in a county the more the cases and death related to COVID.
# The death and the cases in each County related to age needs to be explored further. This would give and insight on 
# the regulations required to be palced in each county during such situations.
# 

# CATEGORICAL PLOT
# 6.Create a categorical plot and interpret the results?
# When considering the variables C.cases(COVID cases),C.deaths and the median age, the categorical plot shows that 
# population below 40 years of age and between 40-60 are at high risk and moderately risk simultaneously, while population 
# above 60 years of age is at low risk.

# In[ ]:





# In[ ]:





# In[ ]:




