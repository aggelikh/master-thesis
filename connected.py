from operator import itemgetter
import numpy as np
import networkx as nx
import pandas as pd
from networkx.algorithms.bipartite import biadjacency_matrix
import sklearn.model_selection
import matplotlib.pyplot as plt


# create the biadjacency matrix, creating a graph with data's edges
def to_adjacency_matrix(data):
    g = nx.Graph()
    g.add_edges_from(data)
    partition_0 = set(map(itemgetter(0), data))
    # partition_1 = set(map(itemgetter(1), data))
    return biadjacency_matrix(g, partition_0).toarray()
    # return biadjacency_matrix(g, partition_0).toarray(), partition_0, partition_1


df = pd.read_csv("final.csv", sep=" ", header=1, names=["Tags", "Users"])

# create train data set as  90% of df and test 10%!!!
train, test = sklearn.model_selection.train_test_split(df, test_size=0.1)

# change columns order so to have users first
train = train[['Users', 'Tags']]
test = test[['Users', 'Tags']]

# Build your graph. Note that we use the Graph function to create the graph, that is an undirected graph
B = nx.Graph()
B.add_nodes_from(test['Users'].values.tolist(), bipartite=0)
B.add_nodes_from(test['Tags'].values.tolist(), bipartite=1)
B.add_edges_from(list(zip(test['Users'].values.tolist(), test['Tags'].values.tolist())))
largest_cc = max(nx.connected_components(B), key=len)
print(len(largest_cc))
print(nx.number_connected_components(B))
# nx.draw(B, with_labels=True)
# plt.show()
