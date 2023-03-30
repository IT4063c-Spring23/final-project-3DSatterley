#!/usr/bin/env python
# coding: utf-8

# # Mass moves mass?📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
#  My question is <b>can ML help predict the greatest body weight to maximize the greatest weight to be lifted within powerlifting/ weight lifting?</b>  

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# In weightlifting, everyone knows 'mass moves mass' meaning people who weight more usually move more weight. However, for the people who want to stay in lower weight classes or just maximize their weight classes they ask themeselves: what is the greatest weight for my body to maximize my lifts without gaining crazy amount of weight or going out of my weight class?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# The more mass a person is, the greater the weight that can be lifted. However I do want to see how the weight lifted compares within a weight class. For example a weight class of 180lbs-190lbs. I could imagine that the people who weight 190lb would lift heavier, but there would be some outliers who are lighter who could lift more weight.
# 
# Some graphs I would highlight are a line graph between weight and weight lifted for the prediction vs the outcome.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 
# The three data sources that I will be using for this project will be the openpowerlifting.csv file (*Had to remove from my data because it exceeded github's size. I have it on my laptop downloads*), the website https://exrx.net/Testing/WeightLifting/DeadliftStandards on all main lifts (Squat, deadlift, and bench), and my own data I have recorded from my lifts as well. I will add my personal data to the powerlifting data to see how my own stats compare to other powerlifters within my weight range. Within the website, I will use the tables to determine where the powerlifters fall into regarding their weight and lifts (If they are itermidiate, advanced, elite, etc.)

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# I could not get my imports to work, however I would import pandas and beautiful soup to retreieve my data.
# 
# I would download the openpowerlifting.csv file from Kaggle using opendatasets
# 
# To open the text file I would use the with open function in read mode.
# 
# To read the tables off the website "https://exrx.net/Testing/WeightLifting/DeadliftStandards" I would use beautiful soup and scrape the tables off the url. I would do this for squat, deaadlift, and bench and combine all the tables using concat.
# 
# 

# # Loading Data

# In[85]:


import pandas as pd
import opendatasets as od
import requests
from bs4 import BeautifulSoup
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")


# In[27]:


dataset_url = 'https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database'
od.download(dataset_url, data_dir='./data')


# In[65]:


powerlifting_df = pd.read_csv('./data/powerlifting-database/openpowerlifting.csv')
powerlifting_df.head()


# In[66]:


personal_df = pd.read_table('./data/personal_data.txt', sep=',')
personal_df


# In[30]:


urls = [('https://exrx.net/Testing/WeightLifting/BenchStandardsKg'), ('https://exrx.net/Testing/WeightLifting/DeadliftStandardsKg'), ('https://exrx.net/Testing/WeightLifting/SquatStandardsKg')]

# create df
standard_df = pd.DataFrame()

for url in urls:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find('table')
    standard_df = pd.read_html(str(table))[0]
    # Merge with some df 
    
standard_df


# In[67]:


final_df = pd.concat((powerlifting_df,personal_df))
final_df


# # EDA

# ## Correlation

# First wanted to have a correlation matrix of all the varaibles before I reduced the columns just to see:

# In[73]:


final_df_corr = final_df.corr()
final_df_corr


# In[75]:


sns.heatmap(final_df_corr)


# ## Cleaning Columns

# I know for my project I will not need the meet or any real information other than their best lifts (the weight that was actually lifted and not the fails), their total weights from all their lifts, their body weight, possibly their weight class and the event has to include all three lifts in order to get the most precise lift totals. Wilks is a value that determines the 'best lifter' baised on calculations to compare everyone even in different weight groups and sex's. 

# In[76]:


final_df = final_df[['Name','Sex','Event','BodyweightKg','WeightClassKg','Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Wilks']]
final_df = final_df[final_df['Event'] == 'SBD']
final_df


# In[77]:


display(final_df.shape)
display(final_df.info())
display(final_df.describe())


# ## Duplicates

# In[78]:


final_df.duplicated().sum()


# In[79]:


final_df[final_df.duplicated]


# I figured to drop the duplicates thinking they just competed multiple times so they had multiple entries. Even if that is the case, the duplicates are not needed for the findings. So they can all be dropped. 

# In[80]:


final_df.drop_duplicates( inplace= True)


# In[81]:


display(final_df.duplicated().sum())
display(final_df.shape)


# ## Missing Values

# In[82]:


final_df.isna().sum()


# Since I am dealing with weightlifting data, I can not just insert a median or average into missing values (for example median bench press is 275lbs and a 140lb lifter has no values. That 140lb lifter now has 275lb bench which is very un-likely). Because of this, I will drop all of the missing values. 

# In[84]:


final_df.dropna(inplace=True)
display(final_df.isna().sum())
display(final_df.shape)


# ## Outliers

# In[86]:


final_df['BodyweightKg'].plot(kind='box')
plt.show()


# In[87]:


q1 = final_df['BodyweightKg'].quantile(.25)
q3 = final_df['BodyweightKg'].quantile(.75)
range = q3-q1

print('Q1: {}, Q3: {}, Range: {} '.format(q1,q3,range))

lower_bound = q1 - 1.5 * range
upper_bound = q3 + 1.5 * range

print('Lower bound: {}, Upper bound: {}'.format(lower_bound,upper_bound))


# In[88]:


final_df[(final_df['BodyweightKg'] < lower_bound) | (final_df['BodyweightKg'] > upper_bound)]


# In[90]:


final_df['BodyweightKg'].plot.hist(bins=40)
bounds = [lower_bound, upper_bound]
for bound in bounds:
    plt.axvline(bound, color='r', linestyle=':')
plt.show()


# After looking at the outliers, I believe I will keep these values. As people can truly be 150kg, especially powerlifters. So I will keep this larger values in. 

# As for the next steps, for ML, I believe I will use the standard_df to help determine if theese lifters are considered above average, intermidiate, elite, etc. from their lifting numbers compared to others in their weight class. 

# # Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[34]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

