import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the data
data = pd.read_csv('seq.csv', header=None)
X = data.iloc[:, :2].values  # First two columns are features
y = data.iloc[:, 2].values   # Last column is labels

# Create color maps
colors = ['#ffcccc', '#ccffcc', '#cce5ff']  # Light colors for decision boundaries
cmap_light = ListedColormap(colors)
colors_bold = ['#ff0000', '#00ff00', '#0000ff']  # Bold colors for data points
cmap_bold = ListedColormap(colors_bold)

# Create a meshgrid to plot decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot the data points
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edcolors='black', linewidth=1)

# Set plot properties
plt.title('Classification Data with Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')

# Optional: Plot test predictions if available
try:
    test_data = pd.read_csv('test_predictions.csv', header=None)
    X_test = test_data.iloc[:, :2].values
    y_pred = test_data.iloc[:, 2].values
    correct = test_data.iloc[:, 3].values  # Assuming 4th column indicates correct/wrong predictions
    
    # Plot correct and wrong predictions
    plt.scatter(X_test[correct == 1][:, 0], X_test[correct == 1][:, 1], 
               c='green', marker='^', label='Correct Predictions', alpha=0.6)
    plt.scatter(X_test[correct == 0][:, 0], X_test[correct == 0][:, 1], 
               c='red', marker='x', label='Wrong Predictions', alpha=0.6)
    plt.legend()
except:
    pass

plt.show()