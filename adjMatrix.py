from operator import itemgetter
import numpy as np
import networkx as nx
import pandas as pd
import scipy
from networkx.algorithms.bipartite import biadjacency_matrix
import sklearn.model_selection
from scipy import sparse
from scipy.sparse import csgraph


# create the biadjacency matrix, creating a graph with data's edges
def to_adjacency_matrix(data):
    g = nx.DiGraph()
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

data = list(zip(*[train[c].values.tolist() for c in train]))

bi_adjacency = to_adjacency_matrix(data)

zero1 = np.zeros((bi_adjacency.shape[0], bi_adjacency.shape[0]), dtype=int)
zero2 = np.zeros((bi_adjacency.shape[1], bi_adjacency.shape[1]), dtype=int)

# np.hstack((np.vstack((zero1, bi_adjacency.transpose())), np.vstack((bi_adjacency, zero2))))
# make tha adjacency matrix for bipartite graph, rectangular!!!
adj_matrix = np.bmat([[zero1, bi_adjacency], [bi_adjacency.transpose(), zero2]])

# useful code for array size, dimensions, sizes in bytes etc
# print("x3 ndim: ", adj_matrix.ndim) == 2
# print("x3 shape:", adj_matrix.shape) == (6338, 6338)
# print("x3 size: ", adj_matrix.size) == 40170244
# print("itemsize:", adj_matrix.itemsize, "bytes") == 4 bytes
# print("nbytes:", adj_matrix.nbytes, "bytes") == 160680976 bytes

data_csr = sparse.csc_matrix(adj_matrix, dtype="int")
print(scipy.sparse.csgraph.connected_components(data_csr, directed=False, return_labels=False))

scipy.sparse.save_npz("adjmatrix.npz", data_csr)
