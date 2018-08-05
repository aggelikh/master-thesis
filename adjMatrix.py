from operator import itemgetter
import numpy as np
import networkx as nx
import pandas as pd
from networkx.algorithms.bipartite import biadjacency_matrix
from sklearn.model_selection import train_test_split


# create the biadjacency matrix, creating a graph with data's edges
def to_adjacency_matrix(data):
    g = nx.DiGraph()
    g.add_edges_from(data)
    partition_0 = set(map(itemgetter(0), data))
    # partition_1 = set(map(itemgetter(1), data))
    return biadjacency_matrix(g, partition_0).toarray()
    # return biadjacency_matrix(g, partition_0).toarray(), partition_0, partition_1


df = pd.read_csv("final.csv", sep=" ", header=1, names=["Tags", "Users"])

# pairnw to train na einai 90% tou df kai to test 10%!!!
train, test = train_test_split(df, test_size=0.1)

# change columns order so to have users first
train = train[['Users', 'Tags']]
test = test[['Users', 'Tags']]

data = list(zip(*[train[c].values.tolist() for c in train]))

bi_adjacency = to_adjacency_matrix(data)

zero1 = np.zeros((bi_adjacency.shape[0], bi_adjacency.shape[0]), dtype=int)
zero2 = np.zeros((bi_adjacency.shape[1], bi_adjacency.shape[1]), dtype=int)

# np.hstack((np.vstack((zero1, bi_adjacency.transpose())), np.vstack((bi_adjacency, zero2))))
adj_matrix = np.bmat([[zero1, bi_adjacency], [bi_adjacency.transpose(), zero2]])
print(adj_matrix.shape)
