import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        
        self.lr = lr  # Learning rate
        self.epochs = epochs  # Number of training iterations
        self.reg_const = reg_const  # Regularization constant
        self.n_class = n_class  # Number of classes
        self.w = 0  # Weights (initialized in train)

    def calc_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray):
        
        N, D = X_batch.shape  # N is the batch size, D is the number of features
        grad_w = np.zeros_like(self.w)  # Initialize gradients to zero
        hinge_loss = 0  # Initialize hinge loss

        for i in range(N):
            x_i = X_batch[i]  # Current data point
            y_i = y_batch[i]  # Current data point's label
            
            dotproductresults = np.dot(self.w, x_i)  # Compute dot product
            correct_class_score = dotproductresults[y_i]  # Score of the correct class
            
            for j in range(self.n_class):
                if j == y_i:
                    continue  # Skip the correct class
                
                margin = dotproductresults[j] - correct_class_score + 1  # Compute margin
                
                if margin > 0:  # Margin violation
                    grad_w[j] += x_i  # Update gradient for the incorrect class
                    grad_w[y_i] -= x_i  # Update gradient for the correct class
                    hinge_loss += margin  # Add the margin to hinge loss

        grad_w /= N  # Average gradient over the mini-batch
        grad_w += self.reg_const * self.w  # Add regularization to the gradient
        hinge_loss += 0.5 * self.reg_const * np.sum(self.w ** 2)  # Regularization loss
        
        return grad_w, hinge_loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray, batch_size=32):
        
        N, D = X_train.shape  # N is the number of training examples, D is the number of features
        self.w = np.random.randn(self.n_class, D) * 0.01  # Initialize weights with small random values
        hinge_loss_history = []  # List to store hinge loss per epoch

        for epoch in range(self.epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)  # Shuffle the training data
            X_train, y_train = X_train[indices], y_train[indices]  # Apply shuffle

            total_loss = 0  # Initialize total loss for the epoch

            for i in range(0, N, batch_size):  # Iterate through mini-batches
                X_batch = X_train[i:i + batch_size]  # Get current mini-batch of data
                y_batch = y_train[i:i + batch_size]  # Get corresponding mini-batch labels
                
                grad_w, batch_loss = self.calc_gradient(X_batch, y_batch)  # Calculate gradient and loss for batch
                
                self.w -= self.lr * grad_w  # Update weights using the gradient
                total_loss += batch_loss  # Accumulate total loss for this epoch

            hinge_loss_history.append(total_loss / (N // batch_size))  # Average loss for this epoch

        return hinge_loss_history  # Return the history of hinge losses over epochs

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        
        return np.argmax(np.dot(X_test, self.w.T), axis=1)  # Return class with the max score  
