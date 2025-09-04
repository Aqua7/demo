import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Synthetic Data Generation ---

def generate_data(num_samples=1000):
    """
    Generate synthetic data for the task:
    Inputs: [x_target, y_target, current_angle]
    Output: delta_angle (angle adjustment to face the target)
    """

    # Random target positions in range [-1, 1]
    x_target = np.random.uniform(-1, 1, num_samples)
    y_target = np.random.uniform(-1, 1, num_samples)

    # Current angles in radians [-pi, pi]
    current_angle = np.random.uniform(-np.pi, np.pi, num_samples)

    # Calculate angle from origin to target
    desired_angle = np.arctan2(y_target, x_target)

    # Calculate angle difference delta (desired - current)
    delta_angle = desired_angle - current_angle

    # Normalize delta_angle to [-pi, pi] to avoid large rotations
    delta_angle = (delta_angle + np.pi) % (2 * np.pi)- np.pi

    # Stack inputs
    X = np.stack([x_target, y_target, current_angle], axis=1)
    y = delta_angle.reshape(-1, 1)

    return X, y

# --- Neural Network Class ---

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, learning_rate=0.01):
        # Initialize weights with small random values
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.Wh = np.random.randn(hidden_size, hidden_size2) * 0.1
        self.bh = np.zeros((1, hidden_size2))
        self.W2 = np.random.randn(hidden_size2, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        # First hidden layer
        self.Z1 = X.dot(self.W1) + self.b1     # Linear
        self.A1 = self.relu(self.Z1)           # Activation
        
        # Second hidden layer
        self.Zh = self.A1.dot(self.Wh) + self.bh
        self.A2 = self.relu(self.Zh)
        
        # Output layer (linear for regression)
        self.Z2 = self.A2.dot(self.W2) + self.b2
        return self.Z2

    def compute_loss(self, y_pred, y_true):
        # Mean Squared Error
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]  # number of samples

        # Output layer gradient
        dZ2 = (2/m) * (y_pred - y_true)  # (m, output_size)

        # Gradients for W2 and b2
        dW2 = self.A2.T.dot(dZ2)          # (hidden_size2, output_size)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backprop into second hidden layer
        dA2 = dZ2.dot(self.W2.T)          # (m, hidden_size2)
        dZ_h = dA2 * self.relu_derivative(self.Zh)  # (m, hidden_size2)

        # Gradients for Wh and bh
        dWh = self.A1.T.dot(dZ_h)         # (hidden_size, hidden_size2)
        dbh = np.sum(dZ_h, axis=0, keepdims=True)

        # Backprop into first hidden layer
        dA1 = dZ_h.dot(self.Wh.T)         # (m, hidden_size)
        dZ1 = dA1 * self.relu_derivative(self.Z1)   # (m, hidden_size)

        # Gradients for W1 and b1
        dW1 = X.T.dot(dZ1)                # (input_size, hidden_size)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.Wh -= self.lr * dWh
        self.bh -= self.lr * dbh
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X_train, y_train, X_val, y_val, epochs=200):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            loss = self.compute_loss(y_pred, y_train)
            self.backward(X_train, y_train, y_pred)

            y_val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val_pred, y_val)

            train_losses.append(loss)
            val_losses.append(val_loss)

            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss:.5f} - Val Loss: {val_loss:.5f}")

        return train_losses, val_losses

    def predict(self, X):
        return self.forward(X)

