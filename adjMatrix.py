import pandas as pd
from sklearn.model_selection import train_test_split
import networkx as nx
from operator import itemgetter
from networkx.algorithms.bipartite import biadjacency_matrix


# create the biadjacency matrix, creating a graph with data's edges
def to_adjacency_matrix(data):
    g = nx.DiGraph()
    g.add_edges_from(data)
    partition_0 = set(map(itemgetter(0), data))
    partition_1 = set(map(itemgetter(1), data))
    return biadjacency_matrix(g, partition_0).toarray(), partition_0, partition_1


df = pd.read_csv("final.csv", sep=" ", header=1, names=["Tags", "Users"])
print(type(df))

# pairnw to train na einai 90% tou df kai to test 10%!!!
train, test = train_test_split(df, test_size=0.1)

# change columns order so to have users first
train = train[['Users', 'Tags']]
test = test[['Users', 'Tags']]

data = list(zip(*[train[c].values.tolist() for c in train]))

bi_adjacency, user_node, tag_node = to_adjacency_matrix(data)
print(bi_adjacency.shape)
print(len(user_node))
print(len(tag_node))

