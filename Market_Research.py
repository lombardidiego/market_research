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
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import manifold, datasets
from sklearn.datasets.samples_generator import make_blobs

import scipy
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
 
import folium
from folium import IFrame

from wordcloud import WordCloud, STOPWORDS
import re

print('Libraries imported.')


# ## Part 1: Preprocessing

# ### Define de address

# In[60]:


address = '101 9 Ave SW, Calgary, AB T2P 1J9, Canada'
radius = 700

geolocator = Nominatim(user_agent="delivery_zones")
location = geolocator.geocode(address)
lat = location.latitude
lng = location.longitude
print('The geograpical coordinate of {} are {}, {}, and the Radius is {} meters.'.format(address, lat, lng, radius))


# ### Get Foursquare information

# In[61]:


CLIENT_ID = 'WNU4QNPM0NESFXS5PNS4FVNI2LXWXV3F5AVCX2UTZDEZYS5H' # your Foursquare ID
CLIENT_SECRET = 'NY50BSQ1QWU3RAXCKQNFBWIEISZOG2KSYPOKXN40JJFPJ0G2' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 1000

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ### Transform data

# In[62]:


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


# In[63]:


url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)

url


# In[64]:


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


# In[65]:


nearby_venues.head()


# ## Part 2: Map

# In[66]:


z_map = folium.Map(location=[lat, lng], zoom_start=13)

folium.features.CircleMarker(
    [lat, lng],
    radius=10,
    color='red',
    popup=address,
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(z_map)

for lat, lng, label in zip(categories_map['Venue Latitude'], categories_map['Venue Longitude'], categories_map['Class']):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(z_map)

    

z_map


# In[ ]:





# ## Part 3: Data Analysis

# ### Distance Analysis

# In[67]:


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


# In[68]:


print ('The average distancie from {} is {}.'.format(address, nearby_attribute['Distance'].mean()))


# In[69]:


nearby_attribute['Distance'].plot(kind='hist', figsize=(12, 6), color=(0.2, 0.4, 0.6, 0.6))

plt.title('Histogram of the distance of venues in a radius of {} meters.'.format(radius)) # add a title to the histogram
plt.ylabel('Number of venues') # add y-label
plt.xlabel('Distance') # add x-label

plt.show()
print ('The average distancie in {} is {}.'.format(address, nearby_attribute['Distance'].mean()))


# ### Naming Analysis

# In[12]:


names = requests.get(url).json()["response"]['groups'][0]['items']

names_list = []

names_list.append([(
            v['venue']['name']) 
    for v in names]) #create a list with: venue name, location, and category

name_attribute = pd.DataFrame([item for venue_list in names_list for item in venue_list])


# In[73]:


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





# In[14]:


nearby_venues.head()


# ### Clustering

# In[15]:


venues_group = nearby_venues[['Venue Category', 'Venue']].groupby(by='Venue Category').count().reset_index()
venues_group.shape


# #### Reduce the number of categories

# In[16]:


venues_group['Venue Category'].unique()

class_restaurant = ['Restaurant', 'Steakhouse', 'American Restaurant', 'Asian Restaurant', 'Mediterranean Restaurant',
                   'Brazilian Restaurant', 'Chinese Restaurant', 'Diner', 'Eastern European Restaurant', 
                    'Vegetarian / Vegan Restaurant', 'Mexican Restaurant', 'Middle Eastern Restaurant', 
                    'Eastern European Restaurant', 'French Restaurant', 'Italian Restaurant', 'Japanese Restaurant', 
                    'Mediterranean Restaurant', 'New American Restaurant', 'Seafood Restaurant', 'Burger Joint']
class_coffe = ['Breakfast Spot', 'Café', 'Deli / Bodega', 'Coffee Shop', 'Bakery']
class_bar = ['Pub', 'Brewery', 'Cocktail Bar',  'Coffee Shop', 'Lounge', 'Gay Bar', 'Gastropub', 'Hookah Bar', 
             'Rock Club', 'Wine Bar']
class_art = ['Bookstore', 'Theater', 'Performing Arts Venue', 'Indie Movie Theater', 'Museum', 'Art Gallery', 'Library']
class_fitness = ['Yoga Studio', 'Gym', 'Gym / Fitness Center']
class_store = ['Department Store', 'Grocery Store', 'Market', 'Park', 'Plaza', 'Shoe Store', 'Shopping Mall', 'Wine Shop']


# In[17]:


def __condition__ (x):
    for restaurant in class_restaurant:
        if x in class_restaurant:
            return 'Restaurant'
        elif x in class_coffe:
            return 'Coffe'
        elif x in class_bar:
            return 'Bar'
        elif x in class_art:
            return 'Art'
        elif x in class_fitness:
            return 'Fitness'
        elif x in class_store:
            return 'Store'
        else:
            return x


# In[18]:


venues_group['Class'] = venues_group.apply(lambda x: __condition__(x["Venue Category"]), axis=1)


# In[19]:


venues_group = venues_group[['Class', 'Venue']]
venues_group = venues_group.groupby(by='Class').sum()


# In[20]:


venues_group.shape


# #### Plotting the venues by categories

# In[21]:


w = venues_group.sort_values(by='Venue', ascending=False)

w.plot(kind='bar', figsize=(12, 6), color=(0.2, 0.4, 0.6, 0.6))

plt.xlabel('') # add to x-label to the plot
plt.ylabel('Number of venues', fontsize=12) # add y-label to the plot
plt.title('Venues by categories in {}.'.format(address), fontsize=15) # add title to the plot
plt.xticks(ha='right', rotation=55, fontsize=16)
plt.yticks(fontsize=12)

plt.show()


# #### Map by Category

# In[22]:


categories_map = nearby_venues
categories_map.shape


# In[23]:


categories_map['Class'] = categories_map.apply(lambda x: __condition__(x["Venue Category"]), axis=1)
categories_map.head()


# In[24]:


categories_map['Class'].unique()


# In[25]:


c1 = categories_map.loc[categories_map['Class']=='Restaurant']
c2 = categories_map.loc[categories_map['Class']=='Bar']
c3 = categories_map.loc[categories_map['Class']=='Coffe']
c4 = categories_map.loc[categories_map['Class']=='Hotel']
c5 = categories_map.loc[categories_map['Class']=='Art']
c6 = categories_map.loc[categories_map['Class']=='Fitness']
c7 = categories_map.loc[categories_map['Class']=='Store']


# In[26]:


c_map = folium.Map(location=[lat, lng], zoom_start=13)

folium.features.CircleMarker(
    [lat, lng],
    radius=10,
    color='red',
    popup=address,
    fill = True,
    fill_color = 'red',
    fill_opacity = 0.6
).add_to(c_map)

for lat, lng, label in zip(c1['Venue Latitude'], c1['Venue Longitude'], c1['Class']):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(c_map)
    
for lat, lng, label in zip(c2['Venue Latitude'], c2['Venue Longitude'], c2['Class']):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='green',
        popup=label,
        fill = True,
        fill_color='green',
        fill_opacity=0.6
    ).add_to(c_map)
    
    
for lat, lng, label in zip(c3['Venue Latitude'], c3['Venue Longitude'], c3['Class']):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='orange',
        popup=label,
        fill = True,
        fill_color='orange',
        fill_opacity=0.6
    ).add_to(c_map)    

    
text = 'This is the end of this project for now'

iframe = folium.IFrame(text, width=700, height=450)
popup = folium.Popup(iframe, max_width=3000)

Text = folium.Marker(location=[lat,lng], popup=popup,
                     icon=folium.Icon(icon_color='green'))
c_map.add_child(Text)    
    
    
c_map

