#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


df =pd.read_csv("D:\Live.csv", delimiter = ',')


# In[14]:


df.shape


# In[15]:


df.head()


# In[16]:


df.info()


# In[17]:


df.isnull().sum()


# In[18]:


df.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)


# In[19]:


df.info()


# In[20]:


df.describe()


# In[21]:


df['status_id'].unique()


# In[22]:


len(df['status_id'].unique())


# In[23]:


df['status_published'].unique()


# In[24]:


len(df['status_published'].unique())


# In[25]:


df['status_type'].unique()


# In[26]:


len(df['status_type'].unique())


# In[27]:


df.drop(['status_id', 'status_published'], axis=1, inplace=True)


# In[28]:


df.info()


# In[29]:


df.head()


# In[30]:


X = df

y = df['status_type']


# In[31]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X['status_type'] = le.fit_transform(X['status_type'])
y = le.transform(y)


# In[32]:


X.info()


# In[33]:


X.head()


# In[34]:


cols = X.columns


# In[35]:


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)


# In[36]:


X = pd.DataFrame(X, columns=[cols])


# In[37]:


X.head()


# In[38]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0) 

kmeans.fit(X)


# In[39]:


kmeans.cluster_centers_


# In[40]:


kmeans.inertia_


# In[41]:


labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))


# In[42]:


print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[43]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[47]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2,random_state=0)

kmeans.fit(X)

labels = kmeans.labels_

# check how many of the samples were correctly labeled

correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))




# In[48]:


kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[49]:


kmeans = KMeans(n_clusters=4, random_state=0)

kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_

correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


# In[50]:


import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Encode categorical variables
le = LabelEncoder()
df['status_type'] = le.fit_transform(df['status_type'])

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fit KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Plot clustering results with PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering with PCA')
plt.show()


# In[ ]:




