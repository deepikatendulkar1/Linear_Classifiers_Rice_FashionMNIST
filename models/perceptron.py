import numpy as np

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int,decay: float = 0.95):
        
        self.n_class = n_class
        self.w = 0  # Initialized weights a zero
        self.lr = lr
        self.epochs = epochs
        self.decay = decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        
        N, D = X_train.shape
        self.w = np.random.rand(self.n_class, D + 1)  # Bias term added
        
        # Training loop
        for epoch in range(self.epochs):
            for i in range(N):
                x_i = np.append(X_train[i], 1)  # Adding bias term directly to input
                y_truevalues = y_train[i]

                # Compute predictions (scores for each class)
                dotproductresult = np.dot(self.w, x_i)
                y_dotproductresult = np.argmax(dotproductresult)  # calulating class with maximum score

                # Updating weights if the prediction is incorrect
                if y_dotproductresult != y_truevalues:
                    self.w[y_truevalues] += self.lr * x_i  # Increasing weight for the correct class
                    self.w[y_dotproductresult] -= self.lr * x_i  # Decreasing weight for the wrong class
           # Apply learning rate decay
            self.lr *= self.decay
            print(f"On Epoch {epoch + 1} the Learning rate is = {self.lr:.6f}")


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        
        N, _ = X_test.shape 
        X_test = np.hstack([X_test, np.ones((N, 1))])  # Adding bias term to test data
        dotproductresult = np.dot(self.w, X_test.T)  # Computing dot product for all test data
        return np.argmax(dotproductresult, axis=0)  # Returning the class with highest dot product result 