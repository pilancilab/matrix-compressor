import numpy as np
import faiss
import torch

class FaissKNeighborsCPU:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        if isinstance(X, torch.Tensor):
            X = X.float()
        elif isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        else:
            raise ValueError(f"Unknown type {type(X)} for input X")
        assert isinstance(y, type(X))
        self.index.add(X)
        self.y = y
        
    def predict_neighbors(self, X):
        if isinstance(X, torch.Tensor):
            X = X.float()
        elif isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        else:
            raise ValueError(f"Unknown type {type(X)} for input X")

        distances, indices = self.index.search(X, k=self.k)
        return distances, indices

    def predict(self, X):
        _, indices = self.predict_neighbors(X)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


class FaissKNeighborsGPU:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), X.shape[1])
        self.index.add(X.float())
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.float(), k=self.k)
        votes = self.y[indices]
        predictions = np.array([torch.argmax(torch.bincount(x)) for x in votes])
        return predictions
