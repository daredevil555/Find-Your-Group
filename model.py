# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :]
from sklearn.cluster import KMeans

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
kmeans.fit(X)

# Saving model to disk
pickle.dump(kmeans, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(type(format(model.predict([[1, 1, 1,1,1,1,1,1,1,1,]]))))