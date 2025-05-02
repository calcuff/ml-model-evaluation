import numpy as np

class KNN():
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X):
        X_square = np.sum(np.square(X), axis=1, keepdims=True)  # (num_test, 1)
        X_train_square = np.sum(np.square(self.X_train), axis=1)  # (num_train,)
        cross_term = X @ self.X_train.T  # (num_test, num_train)
        
        dists = np.sqrt(X_square - 2 * cross_term + X_train_square)  # Broadcasting
        return dists

    def predict(self, X, k=1):
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    def predict_labels(self, dists, k):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # Get indices of k smallest distances
            nearest_idxs = np.argpartition(dists[i], k)[:k]
            # Get labels and compute majority
            nearest_labels = self.y_train[nearest_idxs]
            values, counts = np.unique(nearest_labels, return_counts=True)
            y_pred[i] = values[np.argmax(counts)]
            
        return y_pred