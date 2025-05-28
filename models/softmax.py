import numpy as np

class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.w = 0  # Weights initiallization

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def calc_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        
        N, D = X_batch.shape

        # Computing dot product and softmax probabilities
        scores = np.dot(X_batch, self.w.T)  
        probs = self.softmax(scores)  

        
        y_grad_ = np.zeros_like(probs)
        y_grad_[np.arange(N), y_batch] = 1

        # Computing gradient 
        grad_w = np.dot((probs - y_grad_).T, X_batch) / N  

        
        grad_w += 2 * self.reg_const * self.w  

        return grad_w

    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size=32):
        
        N, D = X_train.shape
        self.w = np.random.rand(self.n_class, D) * 0.01  # assigning small random weights 

        for epoch in range(self.epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, N, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                grad_w = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * grad_w  # Gradient update

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels for test data points."""
        scores = np.dot(X_test, self.w.T)
        return np.argmax(scores, axis=1)  # Return class with highest probability
