#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries 
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 


# In[2]:


#path to project
path = r'/Users/angitha/Achievement 6'


# # CLEANING 'COVID_US_COUNTY' DATA

# In[3]:


#open covid_us_county.csv 
cases_deaths = pd.read_csv(os.path.join(path, '02 Data','Original Data','covid_us_county.csv'))


# In[4]:


#total amount of rows and columns
cases_deaths.shape


# In[5]:


#list column names
cases_deaths.columns.to_list()


# In[6]:


#drop dates from the table 
cases_deaths = cases_deaths.drop(columns = ['1/1/23',
 '1/2/23',
 '1/3/23',
 '1/4/23',
 '1/5/23',
 '1/6/23',
 '1/7/23',                                             
 '1/8/23',
 '1/9/23',
 '1/10/23',
 '1/11/23',
 '1/12/23',
 '1/13/23',
 '1/14/23',
 '1/15/23',
 '1/16/23',
 '1/17/23',
 '1/18/23',
 '1/19/23',
 '1/23/23',
 '1/24/23',
 '1/25/23',
 '1/26/23',
 '1/27/23',
 '1/28/23',
 '1/29/23',
 '1/30/23',
 '1/31/23',
 '2/1/23',
 '2/2/23',
 '2/3/23',
 '2/4/23',
 '2/5/23',
 '2/6/23',
 '2/7/23',
 '2/8/23',
 '2/9/23',
 '2/10/23',
 '2/11/23',
 '2/12/23',
 '2/13/23',
 '2/14/23',
 '2/15/23',
 '2/16/23',
 '2/17/23',
 '2/18/23',
 '2/19/23',
 '2/23/23',
 '2/24/23',
 '2/25/23',
 '2/26/23',
 '2/27/23'])


# In[7]:


#list column names
cases_deaths.columns.to_list()


# In[8]:


#check the first rows in the updated cases_deaths table
cases_deaths.head(10)


# In[9]:


#change fips name to county_code
cases_deaths.rename(columns = {'fips':'county_code'}, inplace = True)


# In[10]:


#check the first rows in the updated cases_deaths table
cases_deaths.head(5)


# In[11]:


#check if there is mixed data types 
for col in cases_deaths.columns.tolist():
  weird = (cases_deaths[[col]].applymap(type) != cases_deaths[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (cases_deaths[weird]) > 0:
    print (col)


# In[12]:


#describe the datatypes
cases_deaths.dtypes


# In[13]:


#change the datatype of the county column 
cases_deaths['county'] = cases_deaths['county'].astype(str)


# In[14]:


#change the datatype of the state_code column 
cases_deaths['state_code'] = cases_deaths['state_code'].astype(str)


# In[15]:


#drop values 'AL'
cases_deaths = cases_deaths.drop(cases_deaths[cases_deaths['county_code'] == 'AL'].index)


# In[16]:


#change the datatype of the state_code column 
cases_deaths['county_code'] = cases_deaths['county_code'].astype(float)


# In[17]:


#check if there is mixed data types again
for col in cases_deaths.columns.tolist():
  weird = (cases_deaths[[col]].applymap(type) != cases_deaths[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (cases_deaths[weird]) > 0:
    print (col)


# In[18]:


#find any missing values 
cases_deaths.isnull().sum()


# In[19]:


#drop missing values in the the county_code column 
cases_deaths.dropna(inplace = True)


# In[20]:


#find any missing values 
cases_deaths.isnull().sum()


# In[21]:


#check for any duplicates 
df_dups = cases_deaths[cases_deaths.duplicated()]
df_dups.head(20)


# In[22]:


#check cases_deaths
cases_deaths.head(5)


# In[23]:


cases_deaths.describe()


# # CLEANING 'US_COUNTY' DATA

# In[24]:


#export us_counties.csv 
gender_population = pd.read_csv(os.path.join(path,'02 Data','Original Data','us_county.csv'))


# In[25]:


#overview of the gender_population table 
gender_population.head(5)


# In[26]:


#check counts of each column 
gender_population.count()


# In[27]:


#check the columns and rows of the gender_population table 
gender_population.shape


# In[28]:


#rename fips column as 'county_code'
gender_population.rename(columns = {'fips':'county_code'}, inplace = True)


# In[29]:


#check the datatypes of the table 
gender_population.dtypes


# In[30]:


#check datatypes 
for col in gender_population.columns.tolist():
  weird = (gender_population[[col]].applymap(type) != gender_population[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (gender_population[weird]) > 0:
    print (col)


# In[31]:


#change the datatypes in the state_code 
gender_population['state_code'] = gender_population['state_code'].astype(str)


# In[32]:


#check the datatypes of the table 
gender_population.dtypes


# In[33]:


#check datatypes 
for col in gender_population.columns.tolist():
  weird = (gender_population[[col]].applymap(type) != gender_population[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (gender_population[weird]) > 0:
    print (col)


# In[34]:


#find missing values 
gender_population.isnull().sum()


# In[35]:


#check duplicates in the dataframe
df = gender_population[gender_population.duplicated()]
df


# In[36]:


#check the descriptive statistics 
gender_population.describe()


# In[37]:


#drop values 'AL'
gender_population = gender_population.drop(gender_population[gender_population['county_code'] == 'AL'].index)


# In[38]:


#change the datatypes in the state_code 
gender_population['county_code'] = gender_population['county_code'].astype(float)


# In[39]:


#check the first 10 columns of the table 
gender_population.head(5)


# # MERGING 'CASE_DEATHS' & 'GENDER_POPULATION' TABLES

# In[40]:


#merge cases_deaths and gender_population tables 
cases_deaths_gender = cases_deaths.merge(gender_population, on = 'county_code',indicator=True)


# In[41]:


#Value counts from _merge column
cases_deaths_gender['_merge'].value_counts()


# In[42]:


#columns in the cases_deaths_gender table 
cases_deaths_gender.columns.to_list()


# In[43]:


#drop duplicate columns 
cases_deaths_gender = cases_deaths_gender.drop(columns= ['county_y','state_y','state_code_y','lat_y','long_y','_merge'])


# In[44]:


cases_deaths_gender.head()


# In[45]:


#rename county column
cases_deaths_gender.rename(columns = {'county_x' : 'county'}, inplace = True)


# In[46]:


#rename state column
cases_deaths_gender.rename(columns = {'state_x' : 'state'}, inplace = True)


# In[47]:


#rename lat column
cases_deaths_gender.rename(columns = {'lat_x' : 'lat'}, inplace = True)


# In[48]:


#rename long column
cases_deaths_gender.rename(columns = {'long_x' : 'long'}, inplace = True)


# In[49]:


#rename state_code column
cases_deaths_gender.rename(columns = {'state_code_x' : 'state_code'}, inplace = True)


# In[50]:


#check the new table
cases_deaths_gender.head()


# In[51]:


#assign the states to regions 
result = [] 
northeast = ['Maine', 'New Hampshire', 'Vermont','Massachusetts', 'Rhode Island', 'Connecticut', 'New York', 'Pennsylvania', 'New Jersey']
midwest = ['Wisconsin','Michigan','Illinois','Indiana','Ohio','North Dakota','South Dakota','Nebraska','Kansas','Minnesota','Iowa', 'Missouri']
south = ['Delaware','Maryland', 'District of Columbia', 'Virginia', 'West Virginia','North Carolina','South Carolina', 'Georgia', 'Florida', 'Kentucky', 'Tennessee', 'Mississippi','Alabama','Oklahoma','Texas','Arkansas','Louisiana']
west = ['Idaho', 'Montana','Wyoming','Nevada','Utah','Colorado','Arizona','New Mexico','Alaska','Washington','Oregon','California','Hawaii']

for value in cases_deaths_gender['state']:
    if value in northeast:
        result.append("Northeast")
    elif value in midwest:
        result.append("Midwest")
    elif value in south:
        result.append("South")
    else:
        result.append("West")


# In[52]:


#add a region column containing the result list
cases_deaths_gender['region'] = result


# In[53]:


#check that the region column was added
x = cases_deaths_gender[cases_deaths_gender['region'] == 'West']
x.head(5)


# In[54]:


#export new cases_deaths_gender 
cases_deaths_gender.to_csv(os.path.join(path,'02 Data','Prepared Data','cases_deaths_gender.csv'))


# In[ ]:




