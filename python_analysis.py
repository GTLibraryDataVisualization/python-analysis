#!/usr/bin/env python
# coding: utf-8

# # Statistical Analysis in Python
# 
# This workshop is an introduction to Statistical Analysis in Python
# 
# It is NOT an introduction to Python workshop, I assume a working knowlege of the language
# 
# It is NOT an introduction to statisics workshop
# 
# Rather, this workshop is designed to show you the tools and basic process to get started on your own. It covers:
# 
# * Using Pandas to load in and manipulate data
# * Using numpy for the many tools it has to work with our data
# * Visualizing data with Seaborn
# 
# This will be done by following a basic proccess to analyze data, but why do we do this process?

# In[5]:


# Import the tools we will be using

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Python is great for scripting, this means we can easily and quickly prepare and manipulate data, but data analysis and modeling require some work, this is where Pandas comes in. It gives python a similar power to a language like R designed for statistics.

# ## Understanding the Data

# ### Pandas
# 
# "Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language." - [Pandas Website](pandas.pydata.org)
# 
# Python doesnt support vectors, matricies, or dataframes out of the box, this is what Pandas is for. Its numpy, but better.
# 
# ##### How do we store and represent data?
# Generally, the standard is a Comma-Seperated Values file, or CSV. Excel and Google Sheets can import and export them. As the name suggests, each line is a row, and each column is seperated by a comma.

# In[21]:


# Load in some data

tips_ds = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")

# This loads in our data as a DataFrame object, labeled columns and rows
# A DataFrame is a 2D mutable data structure, so we can view it as a table

print(type(data))
print(data)


# The data we are using is relatively straight forward, its called "tips"
# lets take a look at what we have. Our first row is just an individual patron to this store. So, each of our colums for each patron is:
# * `total_bill`: FLOAT The final bill at this store
# * `tip`: FLOAT the size of the tip that was left
# * `sex`: STRING the sex of the patron
# * `smoker`: STRING if the person smoked or not
# * `day`: STRING the day of the week of the visit
# * `time`: STRING if the visit was for lunch or dinner
# * `size`: INT the size of the group that came in
# 
# A long list of numbers and strings is pretty meaningless to us, so lets see what we info we can gleam from plotting our data.

# In[90]:


print(tips_ds.size)
print(tips_ds.shape)
print(tips_ds.index)
print(tips_ds.columns)
print(tips_ds.dtypes)
print(tips_ds.empty)


# ### Seaborn
# 
# "Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics." - [Seaborn website](https://seaborn.pydata.org)
# 
# Seaborn is designed to work with Pandas. Its dataset-oriented and helps us look for relationships between multiple variables. Its built on top of matplotlib to make good looking vizualizations for the purpose of data analysis.

# In[22]:


sns.relplot(x="total_bill", y="tip", col="time", hue="smoker", style="smoker", size="size", data=tips_ds)


# We packed a lot of information in this plot, breaking it down:
# * along the x axis we have the size of our bill
# * along the y axis we have the size of our tip
# * we have two columns of charts, on the left, people that came in during dinner, on the right, for lunch
# * dots are non-smokers, Xs are smokers
# * size of the dot is the size of the party
# 
# [`seaborn.relplot()`](https://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot) gives us a relational plot that we can make either a scatter plot (default) or a line plot
# 
# At this point we have a rough idea about what our data is, but lets use Pandas again to get some more info out of it.

# ### Descriptive Statistics

# In[60]:


tips_ds.sum()


# We added the numbers correctly, but we ended up concatenating the strings... Lets try it again, one column at a time. To do that we need to be able to access a column.

# In[204]:


tips_ds.total_bill
tips_ds["total_bill"]


# Both of this work in getting data, but we need to use `tips_ds["total_bill"]` if we want to reassign data.

# In[203]:


print("Num patrons: {}".format(tips_ds.total_bill.count()))
print("Total Money (no tips): {}".format(tips_ds.total_bill.sum()))
print("Total Tips: {}".format(tips_ds.tip.sum()))
print("Avg Party Size: {}".format(tips_ds.size.mean()))
print("Bill/Tip covariance: {}".format(tips_ds.total_bill.cov(tips_ds.tip)))


# or we can just be lazy

# In[62]:


tips_ds.describe()


# This formatting is maintained when copy pasting into Microsoft Word
# 
# Pandas has support for:
# * `dataframe.count()` Number of non-null observations
# * `dataframe.sum()` Sum of Values
# * `dataframe.mean()` Mean/Average
# * `dataframe.median()` Median
# * `dataframe.min()` Minimum
# * `dataframe.max()` Maximum
# * `dataframe.std()` Corrected standard deviation
# * `dataframe.var()` Unbiased Variance
# * `dataframe.skew()` Skewness
# * `dataframe.kurt()` Kurtosis
# * `dataframe.quantile()` Sample Quantile
# * `dataframe.cov()` Covariance
# * `dataframe.corr()` Correlation
# 
# * `dataframe.apply()` we write our own function that takes in an nd aray and pass that in

# In[209]:


def square_root(nd_input):
    return np.sqrt(nd_input)
    

print(tips_ds.total_bill)    
print(tips_ds.total_bill.apply(square_root))
# tips_ds["total_bill"].apply(np.sqrt) # this also works


# If we have time based data, Pandas can also do all the calculations for a rolling window, as opposed to just a series of data.
# 
# Theres so much more pandas can calculate, check out the [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/resampling.html#computations-descriptive-stats) and scroll down to the "Computations / descriptive stats" section
# 
# but Pandas lets us do more than just get information out of our data. It lets us manipulate our data, easily removing cols, to reordering, etc. 

# for illustrative purposes, before we start messing with the data, lets make a copy

# ### Messing with the Data

# In[211]:


tips = tips_ds.copy()
tips


# adding a total_charge (total_bill + tips)

# In[212]:


tips.insert(len(tips.columns), "total_charge", tips.total_bill + tips.tip)
tips


# Lets say our data is too big. If we were trying to use this data for machine learning, we'd take a look at variance and covariance and purge what gives us the least amount of information gain. But lets just remove 'time' for now.

# In[213]:


cols = tips.columns.tolist()
print(cols)
cols.remove('time')
print(cols)
tips = tips[cols]
tips


# now lets sort by total_bill size, then party size, then tip

# In[200]:


tips = tips.sort_values(["total_bill","size","tip"], ascending=[True, True, True])


# Finally, lets just grab data about tips non-smoking, female customers
# We also only want to display their total_bill and tip, then lets just limit displaying it to the first 10.

# In[214]:


tips = tips.loc[(tips.smoker == "No") & (tips.sex == "Female"), ['total_bill', 'tip', 'smoker', 'sex']]
tips.head(10)


# Now we know how to manipulate our data. Pandas can do so much more, from merging data sets, reshaping, transposing,  etc. Whenever you need to so something its worth giving the [API reference](https://pandas.pydata.org/pandas-docs/stable/index.html#) a quick scan.
# 
# Pandas can even do some visualizations, but Seaborn is super simple and the little bit of extra verbosity gives you a lot more power.

# ## Visualizations

# At this point we can explore our data, we've even made some visualizations to try and understand our data.
# Relationships between features is a lot harder with a chart, so next we'll explore how to make visualizations to analyze those relationships.
# 
# First off...

# ## Distributions
# 
# TODO: Histogram + Kernel Density Estimations
# 
# TODO: Hexbins(?) + Kernel Density Estimations
# 
# TODO: Pairwise relations

# ## Categorical 
# 
# There's a few things we can do to explore our data's categorys.
# * Scatter plots:
#     * `stripplot()`
#     * `swarmplot()`
# * Distribution plots
#     * `boxplot()`
#     * `violinplot()`
#     * `boxenplot()`
# * Estimate plots
#     * `pointplot()`
#     * `barplot()`
#     * `countplot()`

# ### Scatter plots

# In[205]:


sns.catplot(x="day", y="total_bill", data=tips_ds) # Standard Strip Plot
sns.catplot(x="day", y="total_bill", jitter=False, data=tips_ds) # Removes random X-axis noise
sns.catplot(x="day", y="total_bill", kind="swarm", data=tips_ds) # Swarm Plot
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips_ds) # Recolors to add a new dimension


# Histograms will give us better insight into how many values are in a specific category, Unlike histograms, Scatter plots give us 2+ dimensions of data. We lose some information as now the number of data points in categlories is no longer explicityl stated, but we can see how a different feature is distributed in those categories.

# ### Distribution plots

# In[207]:


sns.catplot(x="day", y="total_bill", kind="box", data=tips_ds)
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips)
sns.catplot(x="total_bill", y="day", hue="sex", kind="violin", data=tips)


# ## Regressions

# In[172]:


sns.regplot(x="total_bill", y="tip", data=tips_ds)
sns.lmplot(x="total_bill", y="tip", data=tips_ds)


# Other than the shape of our charts `sns.regplot()` and `sns.lmplot()` create some similar linear regressions. Thats normal as `sns.regplot()` combines `sns.regplot()` with a `FacetGrid`. This basically means with `sns.lmplot()` we can explore relationships between more than two variables.

# In[174]:


sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips_ds)


# We should make this a bit easier to parse however.

# In[176]:


sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips_ds, markers=["o", "x"], palette="Set1")


# And exploring different variables...

# In[180]:


sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", row="sex", data=tips_ds, markers=["o", "x"], palette="Set1")


# We can use a joint plot to get a bit more information out.

# In[182]:


sns.jointplot(x="total_bill", y="tip", data=tips_ds, kind="reg")


# 
# Sometimes we have data where a linear regression doesnt quite make sense though:
# 
# # TODO: NONLINEAR REGRESSIONS

# #### Extra Resources

# In[ ]:




