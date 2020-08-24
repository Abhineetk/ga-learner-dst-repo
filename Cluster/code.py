# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load Offers

offers = pd.read_excel(path, sheet_name=0)
transactions = pd.read_excel(path,sheet_name=1)

transactions['n'] = 1

df = pd.merge(offers, transactions)
# Load Transactions
df.shape
df.head(5)

# Merge dataframes


# Look at the first 5 rows


# create pivot table
matrix = pd.pivot_table(df, index='Customer Last Name', columns='Offer #', values='n', 
                    fill_value=0)
matrix = matrix.reset_index()
matrix.shape
matrix.iloc[0,:].values[1:].sum()
# replace missing values with 0


# reindex pivot table


# display first 5 rows


# initialize KMeans object
cluster = KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10, random_state=0)

matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
# create 'cluster' column
matrix.shape


# initialize pca object with 2 components
pca = PCA(n_components=2, random_state=0)
matrix['x']= pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y']= pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
# create 'x' and 'y' columns donoting observation locations in decomposed form

clusters = matrix.iloc[:,[0,33,34,35]]

# dataframe to visualize clusters by customer names
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')

# visualize clusters
data = pd.merge(clusters, transactions)
data = pd.merge(offers, data)

# initialzie empty dictionary

champagne = {}
# iterate over every cluster
for val in data.cluster.unique():
    # observation falls in that cluster
    new_df = data[data.cluster == val]
    # sort cluster according to type of 'Varietal'
    counts = new_df['Varietal'].value_counts(ascending=False)
    # check if 'Champagne' is ordered mostly
    if counts.index[0] == 'Champagne':
        # add it to 'champagne'
        champagne[val] = (counts[0])

# get cluster with maximum orders of 'Champagne' 
cluster_champagne = max(champagne, key=champagne.get)
print(cluster_champagne)

# print out cluster number


# empty dictionary
discount = {}

# iterate over cluster numbers
for val in data.cluster.unique():
    # dataframe for every cluster
    new_df = data[data.cluster == val]
    # average discount for cluster
    counts = new_df['Discount (%)'].values.sum() / len(new_df)
    # adding cluster number as key and average discount as value 
    discount[val] = counts
    # average discount for cluster

    # adding cluster number as key and average discount as value 
cluster_discount = max(discount, key=discount.get)
print(cluster_discount)

# cluster with maximum average discount





