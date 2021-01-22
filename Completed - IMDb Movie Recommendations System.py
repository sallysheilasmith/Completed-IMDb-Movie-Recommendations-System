#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


movie_data = pd.read_csv('movies_metadata.csv', low_memory=False)


# In[4]:


movie_data.head()


# In[5]:


movie_data['overview'].head(10)


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[7]:


tfidf_vector = TfidfVectorizer(stop_words='english')


# In[8]:


movie_data['overview'] = movie_data['overview'].fillna('')


# In[9]:


tfidf_matrix = tfidf_vector.fit_transform(movie_data['overview'])


# In[10]:


from sklearn.metrics.pairwise import linear_kernel


# In[11]:


sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[12]:


indices = pd.Series(movie_data.index, index=movie_data['title']).drop_duplicates()
indices[:10]


# In[13]:


def content_based_recommender(title, sim_scores=sim_matrix):
    idx = indices[title]


# In[16]:


def content_based_recommender(title, sim_scores=sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movie_data['title'].iloc[movie_indices]


# In[30]:


content_based_recommender('Home Alone')


# In[ ]:




