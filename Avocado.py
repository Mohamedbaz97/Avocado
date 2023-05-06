#!/usr/bin/env python
# coding: utf-8

# #### In this project, I will use the "Avocado Prices" dataset from Kaggle, which can be found at this link: https://www.kaggle.com/neuromusic/avocado-prices.
# 
# #### The goal of this project is to analyze the trends and patterns in avocado prices in different regions of the United States over time. Specifically, we will explore the following research questions:
# 
# #### 1- How have avocado prices changed over time?
# #### 2- How do avocado prices vary by region?
# #### 3- Is there a relationship between avocado prices and the number of avocados sold?
# #### 4- Are there seasonal patterns in avocado prices?
# #### 5- How do organic and conventional avocado prices compare?
# #### 6- How do avocado prices vary by region and type?
# 
# #### Before we can start analyzing the data, we need to preprocess it to make it suitable for analysis. We will start by importing the necessary libraries and reading in the data:

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

avocado_df = pd.read_csv(r'C:\Users\Baazz\Downloads\avocado.csv')


# #### Next, we will take a look at the structure of the data using the head() and info() methods:

# In[4]:


avocado_df.head()
avocado_df.info()


# #### This will give us an idea of what the data looks like and how it is structured.
# 
# #### We notice that the dataset contains 14 columns and 18,249 rows. The columns include date, average price, total volume, 4046, 4225, 4770, total bags, small bags, large bags, xlarge bags, type, year, and region. We will drop the irrelevant columns such as 4046, 4225 and 4770 since we are only interested in the total volume of avocados and not the specific types.

# In[5]:


avocado_df.drop(['4046', '4225', '4770'], axis=1, inplace=True)


# #### We can start exploring it to answer our research questions.
# 
# ### Research Question 1: How have avocado prices changed over time?
# #### To answer this question, we will first group the data by date and compute the mean average price for each date. We will then plot a line graph to visualize the trend in avocado prices over time.

# In[6]:


avocado_df['Date'] = pd.to_datetime(avocado_df['Date'])
price_by_date = avocado_df.groupby('Date')['AveragePrice'].mean()

plt.plot(price_by_date)
plt.xlabel('Year')
plt.ylabel('Average Price ($)')
plt.title('Average Avocado Prices by Year')
plt.show()


# #### From the graph, we can see that avocado prices have generally increased over time, with some fluctuations. There is a noticeable spike in prices around mid-2017, after which prices started to decline again.

# ### Research Question 2: How do avocado prices vary by region?
# 
# #### To answer this question, we will first group the data by region and compute the mean average price for each region. We will then plot a bar graph to visualize the differences in avocado prices across regions.

# In[7]:


price_by_region = avocado_df.groupby('region')['AveragePrice'].mean()

plt.bar(price_by_region.index, price_by_region)
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Average Price ($)')
plt.title('Average Avocado Prices by Region')
plt.show()


# #### From the graph, we can see that avocado prices are generally higher in the Northeast and West regions, and lower in the Southeast and Midwest regions.

# 
# ### Research Question 3: Is there a relationship between avocado prices and the number of avocados sold?

# #### To answer this question, we will compute the correlation between the average price and total volume of avocados sold. We will also create a scatter plot to visualize the relationship between the two variables.

# In[8]:


corr = avocado_df['AveragePrice'].corr(avocado_df['Total Volume'])

plt.scatter(avocado_df['Total Volume'], avocado_df['AveragePrice'])
plt.xlabel('Total Volume')
plt.ylabel('Average Price ($)')
plt.title(f'Correlation: {corr:.2f}')
plt.show()


# #### From the scatter plot and correlation value, we can see that there is a weak negative correlation between avocado prices and the total volume of avocados sold. This suggests that as the total volume of avocados sold increases, the average price tends to decrease slightly.

# ### Research Question 4: Are there seasonal patterns in avocado prices?
# 
# #### To answer this question, we will group the data by month and compute the mean average price for each month. We will then plot a line graph to visualize the seasonal patterns in avocado prices.

# In[9]:


avocado_df['Month'] = avocado_df['Date'].dt.month
price_by_month = avocado_df.groupby('Month')['AveragePrice'].mean()

plt.plot(price_by_month)
plt.xlabel('Month')
plt.ylabel('Average Price ($)')
plt.title('Average Avocado Prices by Month')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# #### From the graph, we can see that avocado prices tend to be higher during the first half of the year (January to June) and lower during the second half of the year (July to December). This suggests that there are seasonal patterns in avocado prices, which could be related to factors such as supply and demand.

# ### Research Question 5: How do organic and conventional avocado prices compare?
# 
# #### To answer this question, we will first group the data by type (organic or conventional) and compute the mean average price for each type. We will then plot a bar graph to visualize the differences in avocado prices between organic and conventional types.

# In[10]:


price_by_type = avocado_df.groupby('type')['AveragePrice'].mean()

plt.bar(price_by_type.index, price_by_type)
plt.xlabel('Type')
plt.ylabel('Average Price ($)')
plt.title('Average Avocado Prices by Type')
plt.show()


# #### From the graph, we can see that organic avocados are generally more expensive than conventional avocados. This could be due to the higher cost of producing organic avocados, as well as consumer preferences for organic products.

# ### Research Question 6: How do avocado prices vary by region and type?
# 
# #### To answer this question, we will group the data by region and type, and compute the mean average price for each combination. We will then plot a heatmap to visualize the differences in avocado prices across regions and types.

# In[11]:


price_by_region_type = avocado_df.groupby(['region', 'type'])['AveragePrice'].mean().unstack()

plt.imshow(price_by_region_type, cmap='YlGn')
plt.colorbar()
plt.xticks(range(len(price_by_region_type.columns)), price_by_region_type.columns)
plt.yticks(range(len(price_by_region_type.index)), price_by_region_type.index)
plt.title('Average Avocado Prices by Region and Type')
plt.show()


# #### From the heatmap, we can see that organic avocados are generally more expensive than conventional avocados across all regions. We also see some regional differences in avocado prices, with higher prices in the West and Northeast regions for both organic and conventional avocados.

# ## Research Result:
# 
# #### - Avocado prices have generally increased over time, with some fluctuations. 
# #### - Avocado prices are generally higher in the Northeast and West regions, and lower in the Southeast and Midwest regions.
# #### - There is a weak negative correlation between avocado prices and the total volume of avocados sold.
# #### - There are seasonal patterns in avocado prices, with higher prices during the first half of the year.
# #### - Organic avocados are generally more expensive than conventional avocados, and there are some regional differences in avocado prices.

# In[ ]:




