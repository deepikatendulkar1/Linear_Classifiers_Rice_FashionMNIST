import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
            threshold: decision boundary for classification (e.g., 0.5)
        """
        self.w = 0  # weights initialized to zero
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function with numerical stability improvements.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        z = np.clip(z, -550, 550)  # Doing Clipping to the input to prevent overflow
        sigmoidresult = 1 / (1 + np.exp(-z)) 
        return sigmoidresult 

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape  # N: number of examples, D: number of features
        self.w = np.random.rand(D)  # Initializing the weights randomly using numpy function 

        # Gradient Descent
        for epoch in range(self.epochs):
            # Compute predictions
            dotproductresult = np.dot(X_train, self.w) # calculating dot product between inputs and weights
            predictionresult = self.sigmoid(dotproductresult)

            gradient = np.dot(X_train.T, (predictionresult - y_train)) / N # calulating the gradient

            self.w -= self.lr * gradient #updating weights

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        dotproductresult = np.dot(X_test, self.w) # calculating dot product between inputs and weights
        predictions = self.sigmoid(dotproductresult)
        return (predictions >= self.threshold).astype(int)  
