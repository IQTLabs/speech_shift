import numpy as np
from sklearn.decomposition import PCA

"""
Class structure to extend it to stateful aggregators like rnn or transformers
"""

class MeanAggregator:
    def __init_(self):
        pass
    def aggregate(self,embedding_seq):
        """
        Aggregrates an embedding sequence into fixed length vector by averaging
        across the time dimension

        Inputs:
            embedding_seq - a numpy array of shape (1, embedding dim, duration)
        Outputs:
            embedding - fixed length vector of shape (1,embedding dim)
        """
        return np.mean(embedding_seq,axis=-1)

class PCAAggregator:
    def __init__(self):
        self.pca = PCA(n_components=1)
    def aggregate(self,embedding_seq):
        """
        Aggregrates an embedding sequence into fixed length vector by returning
        the first principle component of the data distribution

        Inputs:
            embedding_seq - a numpy array of shape (1, embedding dim, duration)
        Outputs:
            embedding - fixed length vector of shape (1,embedding dim)
        """
        
        self.pca.fit(embedding_seq[0,:,:].T)
        return np.copy(self.pca.components_)

