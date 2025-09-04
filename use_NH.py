#This file can use the Neural Network class and train it or examine  
# a saved version of the network

import numpy as np
import matplotlib.pyplot as plt
import pickle


# here i imported the file that contains neural network diffs 
from Neural_heartv import NeuralNetwork,generate_data

# using a data set to examine or re train the Neural network
X, y = generate_data(6000)

# Train/test split
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# you can still re train the NN saved in pkl file, train is boolean,
# you can modify the NN used for your server.
train=0

if (train):
# Create Neural Network
    nn = NeuralNetwork(input_size=3, hidden_size=24, hidden_size2=16, output_size=1, learning_rate=0.005)

# Train
    train_losses, val_losses = nn.train(X_train, y_train, X_val, y_val, epochs=12000)
# Save Trained network
    with open('last_trained.pkl', 'wb') as f:
        pickle.dump(nn, f)
    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
else:
    # when not training just use the Saved NN value
    with open('last_trained.pkl', 'rb') as h:
        nn = pickle.load(h)


# Evaluate on test data: Compare predicted vs true delta angles
y_pred = nn.predict(X_val)

plt.figure(figsize=(8,6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.xlabel('True Δθ (radians)')
plt.ylabel('Predicted Δθ (radians)')
plt.title('True vs Predicted Angle Adjustments')
plt.plot([-np.pi, np.pi], [-np.pi,np.pi], 'r--')  # Diagonal reference
plt.xlim([-np.pi, np.pi])
plt.ylim([-np.pi, np.pi])
plt.show()
# Attention: when blue points are close to the line, the prediction is OK!
# you can see The network got confused when dealing with -pi and +pi because
# they are closely related :) .