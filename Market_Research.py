#!/usr/bin/env python
# coding: utf-8

# # Market Research
# 
# Express Market Research, with this code you can make a quick investigation of any location, including aspects such as number of premises, distances, and even a previous analysis of names.

# In[1]:


import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json

from geopy.geocoders import Nominatim

import requests
from pandas.io.json import json_normalize

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans

import folium

from wordcloud import WordCloud, STOPWORDS

print('Libraries imported.')


# ## Part 1: Preprocessing

# ### Define de address

# In[2]:


address = 'Hamilton, IN'
radius = 2500

geolocator = Nominatim(user_agent="delivery_zones")
location = geolocator.geocode(address)
lat = location.latitude
lng = location.longitude
print('The geograpical coordinate of {} are {}, {}, and the Radius is {} meters.'.format(address, lat, lng, radius))


# ### Get Foursquare information

# In[3]:


CLIENT_ID = 'WNU4QNPM0NESFXS5PNS4FVNI2LXWXV3F5AVCX2UTZDEZYS5H' # your Foursquare ID
CLIENT_SECRET = 'NY50BSQ1QWU3RAXCKQNFBWIEISZOG2KSYPOKXN40JJFPJ0G2' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 1000

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ### Transform data

# In[4]:


def getNearbyVenues(names, latitudes, longitudes, radius=radius):
    
    venues_list=[]
    for name, lat, lng in (names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[5]:


url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)

url


# In[6]:


results = requests.get(url).json()["response"]['groups'][0]['items'] #get the relevant data

venues_list = []

venues_list.append([(
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results]) #create a list with: venue name, location, and category

nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
nearby_venues.columns = [
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category'] #convert the venues list in a dataframe

print ('There are {} venues, and {} categories.'.format(nearby_venues['Venue'].count(), 
                                                        nearby_venues['Venue Category'].nunique()))


# In[ ]:





# ## Part 2: Map

# In[7]:


venues_map = folium.Map(location=[lat, lng], zoom_start=13)

folium.features.CircleMarker(
    [lat, lng],
    radius=10,
    color='red',
    popup=address,
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(venues_map)

for lat, lng, label in zip(nearby_venues['Venue Latitude'], nearby_venues['Venue Longitude'], nearby_venues['Venue']):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)

venues_map


# In[ ]:





# ## Part 3: Data Analysis

# ### Distance Analysis

# In[8]:


attribute = requests.get(url).json()["response"]['groups'][0]['items']

attribute_list = []

attribute_list.append([(
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['reasons']['items'][0]['summary'],
            v['venue']['location']['distance']) 
    for v in attribute]) #create a list with: venue name, location, and category

nearby_attribute = pd.DataFrame([item for venue_list in attribute_list for item in venue_list])

nearby_attribute.columns = [
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Evaluation',
                  'Distance'] #convert the venues list in a dataframe

nearby_attribute.head()


# In[9]:


print ('The average distancie from {} is {}.'.format(address, nearby_attribute['Distance'].mean()))


# In[10]:


nearby_attribute['Distance'].plot(kind='hist', figsize=(12, 6))

plt.title('Histogram of the distance of venues in a radius of {} meters.'.format(radius)) # add a title to the histogram
plt.ylabel('Number of venues') # add y-label
plt.xlabel('Distance') # add x-label

plt.show()
print ('The average distancie in {} is {}.'.format(address, nearby_attribute['Distance'].mean()))


# ### Naming Analysis

# In[11]:


names = requests.get(url).json()["response"]['groups'][0]['items']

names_list = []

names_list.append([(
            v['venue']['name']) 
    for v in names]) #create a list with: venue name, location, and category

name_attribute = pd.DataFrame([item for venue_list in names_list for item in venue_list])


# In[12]:


text = name_attribute.loc[:,0]


wordcloud = WordCloud(
    width = 3000,
    height = 1200,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:





# In[ ]:




