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

# In[1]:


import pandas as pd
import numpy as np
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score




# In[2]:


dataset_url = 'https://www.kaggle.com/datasets/open-powerlifting/powerlifting-database'
od.download(dataset_url, data_dir='./data')


# In[3]:


powerlifting_df = pd.read_csv('./data/powerlifting-database/openpowerlifting.csv')
powerlifting_df.head()


# In[4]:


personal_df = pd.read_table('./data/personal_data.txt', sep=',')
personal_df


# In[5]:


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


# In[6]:


final_df = pd.concat((powerlifting_df,personal_df))
final_df


# # EDA

# ## Basic Understanding 

# In[7]:


display(final_df.shape)
display(final_df.info())
display(final_df.describe())


# ## Correlation

# First wanted to have a correlation matrix of all the varaibles before I reduced the columns just to see:

# In[8]:


final_df_corr = final_df.corr()
final_df_corr


# In[ ]:


sns.heatmap(final_df_corr)


# ## Histograms

# I wanted to see the skew of data. For data that is not the best, they have negative points. That means the lifter failed that weight attempt. I will use the 'best' categories to tell the sure weight that the lifter lifted successfully. 

# In[9]:


final_df.hist(bins=50, figsize=(20,15))
plt.show()


# ## Cleaning Columns

# I know for my project I will only need actual numerical values such as BodyWeight, each of their best lifts, and the total of their lifts. Wilks is a value that determines the 'best lifter' baised on calculations to compare everyone even in different weight groups and sex's. 

# ![image.png](attachment:image.png)

# In[10]:


final_df = final_df[final_df['Event'] == 'SBD']
final_df = final_df[['BodyweightKg','Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Wilks']]
final_df


# ## Duplicates

# In[11]:


final_df.duplicated().sum()


# In[12]:


final_df[final_df.duplicated]


# I figured to drop the duplicates thinking they just competed multiple times so they had multiple entries. Even if that is the case, the duplicates are not needed for the findings. So they can all be dropped. 

# In[13]:


final_df.drop_duplicates( inplace= True)


# In[14]:


display(final_df.duplicated().sum())
display(final_df.shape)


# ## Missing Values

# In[15]:


final_df.isna().sum()


# Since I am dealing with weightlifting data, I can not just insert a median or average into missing values (for example median bench press is 275lbs and a 140lb lifter has no values. That 140lb lifter now has 275lb bench which is very un-likely). Because of this, I will drop all of the missing values. 

# In[16]:


final_df.dropna(inplace=True)
display(final_df.isna().sum())
display(final_df.shape)


# In[17]:


final_df.head()


# ## Outliers

# In[18]:


final_df['BodyweightKg'].plot(kind='box')
plt.show()


# In[19]:


q1 = final_df['BodyweightKg'].quantile(.25)
q3 = final_df['BodyweightKg'].quantile(.75)
range = q3-q1

print('Q1: {}, Q3: {}, Range: {} '.format(q1,q3,range))

lower_bound = q1 - 1.5 * range
upper_bound = q3 + 1.5 * range

print('Lower bound: {}, Upper bound: {}'.format(lower_bound,upper_bound))


# In[20]:


final_df[(final_df['BodyweightKg'] < lower_bound) | (final_df['BodyweightKg'] > upper_bound)]


# In[21]:


final_df['BodyweightKg'].plot.hist(bins=40)
bounds = [lower_bound, upper_bound]
for bound in bounds:
    plt.axvline(bound, color='r', linestyle=':')
plt.show()


# After looking at the outliers, I believe I will keep these values. As people can truly be 150kg, especially powerlifters. So I will keep this larger values in. 

# # ML

# I have had a couple of ML classes under my belt. Professor used some ML processes that are a little different from what I am used to. I am using the intel I have learned previous for my project. I am going to use linear regression to predict the Wilks model (how preficient a lifter is) from the lifters body weight and totals. 

# ## Splitting dataset

# In[22]:


X = final_df.drop(columns=['Wilks'], axis = 1)
Y = final_df['Wilks']


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)


# ## Train model

# In[24]:


linear_model = LinearRegression().fit(x_train, y_train)


# In[25]:


print("Training_score : " , linear_model.score(x_train, y_train))


# In[26]:


predictors = x_train.columns
print(predictors)


# In[27]:


print(linear_model.intercept_)


# In[28]:


coef = pd.Series(linear_model.coef_, predictors).sort_values()

print(coef)


# In[29]:


y_pred = linear_model.predict(x_test)


# ## Results

# In[30]:


results_df = pd.DataFrame({'predicted': y_pred, 'actual': y_test, 'difference': (y_pred - y_test)})

results_df.head(10)


# In[31]:


fig, ax = plt.subplots(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=.2)
plt.show()


# In[ ]:


print("Testing R2 score: ", r2_score(y_test, y_pred),"RMSE score: ",np.sqrt(mean_squared_error(y_test,y_pred)))


# ## Simple Linear Regression

# In[44]:


X = final_df[['BodyweightKg','TotalKg']]
y = final_df['Wilks']

wilks_model = LinearRegression()
wilks_model.fit(X, y)

y_pred = wilks_model.predict(X)


# In[45]:


x_surf, y_surf = np.meshgrid(
  np.linspace(final_df.BodyweightKg.min(), final_df.BodyweightKg.max(), 100),
  np.linspace(final_df.TotalKg.min(), final_df.TotalKg.max(), 100)
)
surfaceX = pd.DataFrame({'BodyweightKg': x_surf.ravel(), 'TotalKg': y_surf.ravel()})
predictedWilksForSurface=wilks_model.predict(surfaceX)

predictedWilksForSurface=np.array(predictedWilksForSurface)


fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_df['BodyweightKg'],final_df['TotalKg'],final_df['Wilks'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, predictedWilksForSurface.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('BodyweightKg')
ax.set_ylabel('TotalKg')
ax.set_zlabel('Wilks')
plt.show()


# # Fun Facts about the data

# The largest wilks value, which means is the strongest lifter in the whole data set, is not the heaviest lifter. I always thought the more mass someone would be, means the stronger the person. This wilks value shows that body weight does not equal strength.

# In[43]:


final_df.loc[final_df['BodyweightKg'].idxmax()]


# In[42]:


final_df.loc[final_df['Wilks'].idxmax()]


# # Fun facts with my personal Data

# I wanted to use my linear regression model to predict the Wilks value given the body weight and totalKg lifted to predict my Wilks value. I entered my body weight and my total, and I recieved a 319. Which I am not really the happiest with, however I will take it. 

# ![image.png](attachment:image.png)

# In[46]:


wilks_model.predict([[86.2,435.5]])


# # Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# ChatGPT and your notes/ in class sandbox as well as previous labs and assignments

# Your in-class sandbox notebooks as well as class notes, ChatGBT, stackoverflow

# In[ ]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

