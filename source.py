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
# The three data sources that I will be using for this project will be the openpowerlifting.csv file, the website https://exrx.net/Testing/WeightLifting/DeadliftStandards on all main lifts (Squat, deadlift, and bench), and my own data I have recorded from my lifts as well. I will add my personal data to the powerlifting data to see how my own stats compare to other powerlifters within my weight range. Within the website, I will use the tables to determine where the powerlifters fall into regarding their weight and lifts (If they are itermidiate, advanced, elite, etc.)

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# I could not get my imports to work, however I would import pandas and beautiful soup to retreieve my data.
# 
# To open the csv file I would use "pd.read_csv('openpowerlifting.csv')". 
# 
# To open the text file I would use the with open function in read mode.
# 
# To read the tables off the website "https://exrx.net/Testing/WeightLifting/DeadliftStandards" I would use beautiful soup and scrape the tables off the url. I would do this for squat, deaadlift, and bench and combine all the tables using concat.
# 
# 

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[2]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

