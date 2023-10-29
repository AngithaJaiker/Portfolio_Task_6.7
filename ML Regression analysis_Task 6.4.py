#!/usr/bin/env python
# coding: utf-8

# # ML_Regression analysis_Part-1

# This script contains the following:
# 1. Importing libraries and data
# 2. Data cleaning
# 3. Data prep for regression analysis
# 4. Regression analysis

# In[1]:


#1. Importing libraries and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# This option ensures that the graphs you create are displayed within the notebook without the need to "call" them specifically.

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


path = r'/Users/angitha/Achievement 6.4/'


# In[4]:


df = pd.read_csv(os.path.join(path, '02.Data','Original data','cases_deaths_gender_cleaned.csv'))


# In[5]:


#2. Data cleaning- Imported cleaned dataset
df.columns


# In[6]:


df.head(20)


# In[7]:


df.shape


# In[8]:


# Check for missing values

df.isnull().sum()

# No missing values to handle


# In[9]:


#Check duplicates
dups = df.duplicated()


# In[10]:


dups.shape # No dups


# In[11]:


#Extreme value checks
sns.distplot(df['median_age'], bins=25) #No extreme values or outliers


# In[12]:


df['median_age'].describe()


# In[13]:


#Extreme value checks
sns.distplot(df['C.cases'], bins=25) #No extreme values or outliers


# In[14]:


df['C.cases'].describe()


# In[15]:


#3. Data prep for regression analysis
# Create a scatterplot using matplotlib for another look at how the chosen variables plot against each other.
df.plot(x = 'median_age', y='C.cases',style='o') # The style option creates a scatterplot; without it, we only have lines.
plt.title('Median age vs COVID cases')  
plt.xlabel('median_age')  
plt.ylabel('C.cases')  
plt.show()


# In[16]:


# Reshape the variables into NumPy arrays and put them into separate objects.

X = df['median_age'].values.reshape(-1,1)
y = df['C.cases'].values.reshape(-1,1)


# In[17]:


X


# In[18]:


y


# In[19]:


# Split data into a training set and a test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[20]:


#4.Regression analysis

# Create a regression object.
regression = LinearRegression()  # This is the regression object, which will be fit onto the training set.


# In[21]:


# Fit the regression object onto the training set.

regression.fit(X_train, y_train)


# In[22]:


# Predict the values of y using X.

y_predicted = regression.predict(X_test)


# In[23]:


# Create a plot that shows the regression line from the model on the test set.

plot_test = plt
plot_test.scatter(X_test, y_test, color='gray', s = 15)
plot_test.plot(X_test, y_predicted, color='red', linewidth =3)
plot_test.title('Median age vs COVID cases (Test set)')
plot_test.xlabel('median_age')
plot_test.ylabel('C.cases')
plot_test.show()


# In[24]:


# Create objects that contain the model summary statistics.

rmse = mean_squared_error(y_test, y_predicted) # This is the mean squared error
r2 = r2_score(y_test, y_predicted) # This is the R2 score. 


# In[25]:


# Print the model summary statistics. This is where you evaluate the performance of the model.

print('Slope:' ,regression.coef_)
print('Mean squared error: ', rmse)
print('R2 score: ', r2)


# In[26]:


y_predicted


# In[27]:


# Create a dataframe comparing the actual and predicted values of y.

data = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predicted.flatten()})
data.head(30)


# In[28]:


#Compare how the regression fits the training set
# Predict.
y_predicted_train = regression.predict(X_train) # This is predicting X_train!


# In[29]:


rmse = mean_squared_error(y_train, y_predicted_train)
r2 = r2_score(y_train, y_predicted_train)


# In[30]:


print('Slope:' ,regression.coef_)
print('Mean squared error: ', rmse)
print('R2 score: ', r2)


# In[31]:


# Visualizing the training set results.

plot_test = plt
plot_test.scatter(X_train, y_train, color='green', s = 15)
plot_test.plot(X_train, y_predicted_train, color='red', linewidth =3)
plot_test.title('Median age vs COVID cases (Train set)')
plot_test.xlabel('median_age')
plot_test.ylabel('C.cases')
plot_test.show()


# In[32]:


#Performance improvement after removing outliers
# Clean the extreme values from the "median_age" variable observed during the consistency checks.

df_test = df[df['C.cases'] <=200000] 


# In[33]:


# See how the scatterplot looks without outliers.

df_test.plot(x = 'median_age', y='C.cases', style='o')  
plt.title('Median age vs COVID cases')  
plt.xlabel('median_age')  
plt.ylabel('C.cases')  
plt.show()


# In[34]:


# Reshape again.

X_2 = df_test['median_age'].values.reshape(-1,1)
y_2 = df_test['C.cases'].values.reshape(-1,1)


# In[35]:


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.3, random_state=0)


# In[36]:


# Run and fit the regression.

regression = LinearRegression()  
regression.fit(X_train_2, y_train_2)


# In[37]:


# Predict.

y_predicted_2 = regression.predict(X_test_2)


# In[38]:


rmse = mean_squared_error(y_test_2, y_predicted_2)
r2 = r2_score(y_test_2, y_predicted_2)


# In[39]:


print('Slope:' ,regression.coef_)
print('Mean squared error: ', rmse)
print('R2 score: ', r2)


# In[40]:


# Visualizing the test set results.
plot_test = plt
plot_test.scatter(X_test_2, y_test_2, color='gray', s = 15)
plot_test.plot(X_test_2, y_predicted_2, color='red', linewidth =3)
plot_test.title('Median age vs COVID cases (Test set)')
plot_test.xlabel('median_age')
plot_test.ylabel('C.cases')
plot_test.show()


# In[42]:


data = pd.DataFrame({'Actual': y_test_2.flatten(), 'Predicted': y_predicted_2.flatten()})
data.head(60)


# ANSWERS
# 5.State your hypothesis in a markdown cell within your Jupyter notebook.
#     The population at the age group of 30 to 50 are suceptible to COVID virus infections.
#     
# 10.Write your own interpretation of how well the line appears to fit the data in a markdown cell. 
#     The line shows a negative regression, appears to support the hypothesis, the slope drops slightly as
#     the X value increases marking a decrease in the y value. This means that the COVID cases is less among
#     the population with age 50 and above.
#     
# 13.Include your thoughts on how well the model performed on the test set in a markdown cell.
#    Include any reflections you have on the impact of possible data bias.
#     The model is not a good fit for the dataset. The variables in the dataset has a curved relationship. 
#     The MSE value and R2 score also indicates that the regresson value doesnt represent the datapoints accurately
#     The R2 value which indicates the variance in data is close to zero showing that it is a poor fit. 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




