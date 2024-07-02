#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# # GET THE DATASET

# In[3]:


df=pd.read_csv("u.data",sep='\t')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep='\t',names=columns_name)


# In[7]:


df.head()


# In[8]:


movies_titles=pd.read_csv("u.item",sep="\|",encoding='ISO-8859-1',header=None)


# In[9]:


movies_titles.head()


# In[10]:


movie_titles=movies_titles[[0,1]]


# In[11]:


movie_titles


# In[12]:


movie_titles.columns=['item_id','title']


# In[13]:


movie_titles.head()


# In[14]:


df=pd.merge(df,movie_titles,on='item_id')


# # EXPLORATORY Data Analysis

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# In[16]:


df.groupby('title').mean()['rating'].sort_values(ascending=False)


# In[17]:


df.groupby('title').mean()['rating'].sort_values(ascending=False).head()


# In[18]:


df.groupby('title').count()['rating'].sort_values(ascending=False)


# In[19]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[20]:


ratings.head()


# In[21]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# In[22]:


ratings.head()


# In[23]:


ratings.sort_values(by='rating',ascending=False)


# In[24]:


ratings


# # Create Movie Recommendation

# In[25]:


df.head()


# In[26]:


movie_matrix=df.pivot_table(index='user_id',columns='title',values='rating')


# In[27]:


ratings.sort_values('num of ratings',ascending=False).head()


# In[28]:


starwar_user_rating=movie_matrix['Star Wars (1977)']


# In[29]:


starwar_user_rating.head()


# In[30]:


similar_to_starwars=movie_matrix.corrwith(starwar_user_rating)


# In[31]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['Correlation'])


# In[32]:


corr_starwars.dropna(inplace=True)


# In[33]:


corr_starwars.head()


# In[34]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[35]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])
corr_starwars


# In[36]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False)


# # Prediction Function

# In[37]:


def predict_movies(movie_name):
    movie_user_rating=movie_matrix[movie_name]
    similar_to_movie=movie_matrix.corrwith(movie_user_rating)
    
    corr_movie=pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie=corr_movie.join(ratings['num of ratings'])
    corr_movie
    Predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False)
    
    return Predictions


# In[ ]:





# In[38]:


predict_movies('Titanic (1997)').head(n=10)


# In[ ]:




