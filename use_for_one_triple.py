# This file uses the trained neral network  to a triple of inputs from the
# game , it is preferable to make [x,y] in ranges [[-1,1],[-1,1]]
# and theta in range [-pi,pi]

import math
import numpy as np
import matplotlib.pyplot as plt
import pickle


# calling defs :
from Neural_heartv import NeuralNetwork


# example of input form for the nn neural network
neural_in=np.array([1,0.4,0.5*np.pi])

# calling saved NN
with open('last_trained.pkl', 'rb') as h:
    nn = pickle.load(h)

# atan2 returns the angle in radians. The canvas rotation is also in radians.
target_angle = math.atan2(neural_in[1],neural_in[0] )
        
        # Adjust cannon angle to face the target
angle_diff =( target_angle - neural_in[2])
  
# Normalize the angle difference to be within -pi to pi
if angle_diff > math.pi:
    angle_diff -= 2 * math.pi
if angle_diff < -math.pi:
    angle_diff += 2 * math.pi

# Evaluate on test data: Compare predicted vs true delta angles
y_pred = nn.predict(neural_in)

# i printed the values of predicted and true  (Δθ)
print(y_pred,angle_diff)
# using previous plot to show how close to the line.
plt.figure(figsize=(8,6))
plt.scatter(angle_diff, y_pred, alpha=0.5)
plt.xlabel('True Δθ (radians)')
plt.ylabel('Predicted Δθ (radians)')
plt.title('True vs Predicted Angle Adjustments')
plt.plot([-np.pi, np.pi], [-np.pi,np.pi], 'r--')  # Diagonal reference
plt.xlim([-np.pi, np.pi])
plt.ylim([-np.pi, np.pi])
plt.show()