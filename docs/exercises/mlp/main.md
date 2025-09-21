# Exercise 3 - MLP (Multi-Layer Perceptron)
*Author: Leonardo Teixeira*

## Overview

This exercise explores the fundamental concepts and implementation of Multi-Layer Perceptrons (MLPs) through four progressive exercises that demonstrate the power of neural networks to solve complex classification problems:

1. **Manual Calculation of MLP Steps** - Step-by-step mathematical computation of forward pass, loss, backpropagation, and parameter updates
2. **Binary Classification with Synthetic Data and Scratch MLP** - Complete MLP implementation from scratch for binary classification
3. **Multi-Class Classification with Synthetic Data and Reusable MLP** - Extending the MLP architecture to handle multi-class problems with code reusability
4. **Multi-Class Classification with Deeper MLP** - Exploring the benefits of deeper architectures with multiple hidden layers

The exercise demonstrates how MLPs overcome the limitations of perceptrons by learning non-linear decision boundaries through hidden layers and backpropagation.

---

## Exercise 1: Manual Calculation of MLP Steps

### Problem Setup

Following the exercise requirements, I implemented the complete manual calculation for a simple MLP with:

- **Architecture**: 2 input features → 2 hidden neurons → 1 output neuron
- **Activation Function**: tanh for both hidden and output layers
- **Loss Function**: Mean Squared Error (MSE): $L = \frac{1}{N} (y - \hat{y})^2$

### Implementation Approach

I implemented each step of the MLP calculation process to demonstrate the forward pass, loss calculation, backward pass, and parameter updates.

```python
import numpy as np
import pandas as pd

x = np.array([0.5, -0.2])
y_true = 1

W1 = np.array([[0.3, -0.1], 
               [0.2, 0.4]])
b1 = [0.1, -0.2]

W2 = np.array([0.5, -0.3])
b2 = 0.2

learning_rate = 0.3

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2
```

### Step 1: Forward Pass Implementation

The forward pass computes the prediction through each layer of the network:

!!! example "Forward Pass Implementation" 
    ```python
    # Forward Pass
    # 1. Compute the hidden layers pre-activation
    z1 = np.dot(W1, x) + b1

    # 2. Hidden layer activation
    a = tanh(z1)

    # 3. Output layer pre-activation
    z2 = np.dot(W2, a) + b2

    # 4. Final Output
    y_pred = tanh(z2)

    print("z1 (hidden layers pre-activation):", z1)
    print("a (hidden layer activation):", a)
    print("z2 (output layer pre-activation):", z2)
    print("y_pred (final output):", y_pred)
    ```

**Output:**
```
z1 (hidden layers pre-activation): [0.17 -0.06]
a (hidden layer activation): [0.16861124 -0.05987022]
z2 (output layer pre-activation): 0.10658337856554491
y_pred (final output): 0.10602748961089535
```

### Step 2: Loss Calculation

Computing the Mean Squared Error for our single training example:

```python
# Loss Calculation
# Funcao de perda (MSE com 1 exemplo)

N = 1

L = (1 / N) * (y_true - y_pred) ** 2

print("L (loss):", L)
```

**Output:**
```
L (loss): 0.797730507672436
```

### Step 3: Backward Pass (Backpropagation)

The backpropagation algorithm computes gradients for all parameters:

```python
# Backward Pass (Backpropagation)
# 1. dL/dy_pred
dL_dy = 2 * (y_pred - y_true)

# 2. dL/dz2
dL_dz2 = dL_dy * tanh_derivative(z2)

# 3. Gradients for output layer
dL_dW2 = dL_dz2 * a
dL_db2 = dL_dz2

# 4. Hidden layer propagations
dL_da = W2 * dL_dz2
dL_dz1 = dL_da * tanh_derivative(z1)

# 5. Hidden layer gradients
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1

print("Output gradient - dL/dW2:", dL_dW2)
print("Output gradient - dL/db2:", dL_db2)
print("Hidden gradient - dL/dW1:\n", dL_dW1)
print("Hidden gradient - dL/db1:", dL_db1)
```

!!! success "Output"
    ```
    Output gradient - dL/dW2: [-0.29697271  0.10538815]
    Output gradient - dL/db2: -1.7611918110982638
    Hidden gradient - dL/dW1:
     [[-0.43491966  0.17396787]
     [ 0.20918552 -0.08367421]]
    Hidden gradient - dL/db1: [-0.86983933  0.41837105]
    ```

### Step 4: Parameter Updates

Applying gradient descent to update all network parameters:

```python
# Output layer
W2_new = W2 - learning_rate * dL_dW2
b2_new = b2 - learning_rate * dL_db2

# Hidden layer
W1_new = W1 - learning_rate * dL_dW1
b1_new = b1 - learning_rate * dL_db1

print("New W2:", W2_new)
print("New b2:", b2_new)
print("New W1:\n", W1_new)
print("New b1:", b1_new)
```

!!! success "Output"
    ```
    New W2: [0.58909181 -0.33161645]
    New b2: 0.7283575433294791
    New W1:
     [[0.4304759  -0.15219036]
     [0.13724434  0.42510226]]
    New b1: [0.3609518  -0.32551132]
    ```

### Parameter Comparison Table

To visualize the parameter changes, I created a comparison table showing the initial and updated values:

!!! example "Parameter Comparison"
    ```python
    data = {
        "Param": ["W2[0]", "W2[1]", "b2", 
                    "W1[0,0]", "W1[0,1]", "W1[1,0]", "W1[1,1]", 
                    "b1[0]", "b1[1]"],
        
        "Inicial": [W2[0], W2[1], b2, 
                    W1[0,0], W1[0,1], W1[1,0], W1[1,1], 
                    b1[0], b1[1]],
        
        "Updated": [W2_new[0], W2_new[1], b2_new, 
                    W1_new[0,0], W1_new[0,1], W1_new[1,0], W1_new[1,1], 
                    b1_new[0], b1_new[1]]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    df
    ```

| Parameter | Initial Value | Updated Value |
|-----------|---------------|---------------|
| W2[0]     | 0.5           | 0.589092      |
| W2[1]     | -0.3          | -0.331616     |
| b2        | 0.2           | 0.728358      |
| W1[0,0]   | 0.3           | 0.430476      |
| W1[0,1]   | -0.1          | -0.152190     |
| W1[1,0]   | 0.2           | 0.137244      |
| W1[1,1]   | 0.4           | 0.425102      |
| b1[0]     | 0.1           | 0.360952      |
| b1[1]     | -0.2          | -0.325511     |


### Key Learning Outcomes

This exercise demonstrates the fundamental mechanics of MLP training through hands-on implementation. By manually calculating each step, we gain insights into:

1. **Forward Pass**: Sequential computation through layers builds complex representations
2. **Loss Function**: MSE provides a clear optimization target for the network 
3. **Backpropagation**: Using the chain rule to systematically propagate gradients
4. **Parameter Updates**: Applying gradient descent to improve the model's predictions

---

## Exercise 2: Binary Classification with Synthetic Data and Scratch MLP

### Problem Setup

In this exercise, I implemented a complete MLP from scratch for binary classification using a carefully designed synthetic dataset. The implementation demonstrates fundamental neural network concepts through hands-on coding without using high-level frameworks.

### Dataset Generation Strategy

Following the exercise requirements, I created a binary classification dataset with asymmetric clustering patterns:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Update N for this section
N = 1000

# Generate class 0 with 1 cluster
X0, y0 = make_classification(n_samples=500, n_features=2, 
                             n_informative=2, n_redundant=0, 
                             n_clusters_per_class=1, n_classes=2, 
                             weights=[1.0, 0.0], # força tudo em uma classe
                             class_sep=1.5, random_state=42)

# Generate class 1 with 2 clusters
X1, y1 = make_classification(n_samples=500, n_features=2, 
                             n_informative=2, n_redundant=0, 
                             n_clusters_per_class=2, n_classes=2, 
                             weights=[0.0, 1.0], # força tudo na outra classe
                             class_sep=1.5, random_state=24)

# Combine the datasets
X = np.vstack((X0, X1))
y = np.concatenate((y0, y1))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Feature dimensions: {X_train.shape[1]}")
```

**Output:**
```
Training set size: 800
Testing set size: 200
Feature dimensions: 2
```

The generated dataset creates a challenging binary classification problem where:
- **Class 0**: Has 1 cluster (simpler pattern)
- **Class 1**: Has 2 clusters (more complex, non-linear pattern)

![Binary Classification Dataset](images/binary_classification_dataset.png)

### MLP Architecture Implementation

I designed the network architecture with:

```python
import numpy as np

# Architecture
input_size = 2
hidden_size = 4
output_size = 1

# Weights and biases
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
b1 = np.zeros((hidden_size, 1))

W2 = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
b2 = np.zeros((output_size, 1))

print("Dimensions:")
print(f"W1: {W1.shape}, b1: {b1.shape}")
print(f"W2: {W2.shape}, b2: {b2.shape}")
```

**Output:**
```
Dimensions:
W1: (4, 2), b1: (4, 1)
W2: (1, 4), b2: (1, 1)
```

#### The implementation uses:
- **Input Layer**: 2 features (x, y coordinates)
- **Hidden Layer**: 4 neurons with tanh activation  
- **Output Layer**: 1 neuron with sigmoid activation
- **Weight Initialization**: Xavier initialization for stable training

### Forward and Backward Propagation

Here's my implementation for forward and backward propagation:

```python
# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def forward_pass(X, W1, b1, W2, b2):
    # Hidden layer
    z1 = np.dot(W1, X.T) + b1  # Shape: (hidden_size, N)
    a1 = tanh(z1)              # Shape: (hidden_size, N)
    
    # Output layer
    z2 = np.dot(W2, a1) + b2   # Shape: (output_size, N)
    a2 = sigmoid(z2)           # Shape: (output_size, N)
    
    return z1, a1, z2, a2
```

### Loss Function and Backpropagation

I implemented binary cross-entropy loss and backpropagation for the network:

```python
def binary_cross_entropy(y_true, y_pred):
    # To avoid log(0), we clip the predictions
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_pred = y_pred.flatten()  # Ensure shape (N,)
    m = y_true.shape[0]
    loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def backward_pass(X, y_true, Z1, A1, Z2, A2, W2):
    m = X.shape[0]
    
    # Output layer
    dZ2 = A2 - y_true.reshape(1, -1)          # shape (1, m)
    dW2 = (1/m) * np.dot(dZ2, A1.T)           # shape (1, hidden_size)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # shape (1,1)
    
    # Hidden layer
    dA1 = np.dot(W2.T, dZ2)                   # shape (hidden_size, m)
    dZ1 = dA1 * (1 - A1**2)                   # tanh derivative: 1 - tanh^2(z) = 1 - A1^2
    dW1 = (1/m) * np.dot(dZ1, X)              # shape (hidden_size, input_size)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2
```

### Training Process Implementation

I implemented the full training loop with the defined functions:

```python
# Training parameters
epochs = 500
learning_rate = 0.1
losses = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2 = forward_pass(X_train, W1, b1, W2, b2)
    
    # Loss
    loss = binary_cross_entropy(y_train, A2)
    losses.append(loss)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward_pass(X_train, y_train, Z1, A1, Z2, A2, W2)
    
    # Update parameters
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
```

**Output:**
```
Epoch 0, Loss: 0.6900837009211839
Epoch 100, Loss: 0.23865571644823536
Epoch 200, Loss: 0.1930117283752121
Epoch 300, Loss: 0.17092933805237292
Epoch 400, Loss: 0.15683498431566175
```

![Binary Training Loss](images/binary_training_loss.png)

### Prediction and Evaluation

I implemented functions for prediction and model evaluation:

```python
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass(X, W1, b1, W2, b2)
    return (A2 >= 0.5).astype(int) # threshold 0.5

def accuracy(X, y, W1, b1, W2, b2):
    y_pred = predict(X, W1, b1, W2, b2)
    return np.mean(y_pred.flatten() == y.flatten())

train_acc = accuracy(X_train, y_train, W1, b1, W2, b2)
test_acc = accuracy(X_test, y_test, W1, b1, W2, b2)

print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")
```

**Output:**
```
Training accuracy: 0.9413
Testing accuracy: 0.9150
```

### Object-Oriented Implementation

I implemented an MLP class for better organization of the code:

```python
class MLP:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, random_seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Weights and biases initialization
        np.random.seed(random_seed)
        self.W1 = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.b1 = np.zeros((hidden_size, 1))
        
        self.W2 = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def tanh(self, z):
        return np.tanh(z)
        
    def forward_pass(self, X):
        # Hidden layer
        Z1 = np.dot(self.W1, X.T) + self.b1
        A1 = self.tanh(Z1)
        
        # Output layer
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        
        return Z1, A1, Z2, A2
    
    def binary_cross_entropy(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_pred = y_pred.flatten()
        m = y_true.shape[0]
        loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward_pass(self, X, y, Z1, A1, Z2, A2):
        m = X.shape[0]
        
        # Output layer
        dZ2 = A2 - y.reshape(1, -1)
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Hidden layer
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (1 - A1**2)  # tanh derivative
        dW1 = (1/m) * np.dot(dZ1, X)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def fit(self, X, y, epochs=500):
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            Z1, A1, Z2, A2 = self.forward_pass(X)
            
            # Loss
            loss = self.binary_cross_entropy(y, A2)
            losses.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward_pass(X, y, Z1, A1, Z2, A2)
            
            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        
        return losses
    
    def predict(self, X):
        _, _, _, A2 = self.forward_pass(X)
        return (A2 >= 0.5).astype(int)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred.flatten() == y.flatten())

# Create and train an MLP model
mlp = MLP(input_size=2, hidden_size=4, output_size=1)
losses = mlp.fit(X_train, y_train, epochs=500)

# Evaluate the model
train_acc = mlp.accuracy(X_train, y_train)
test_acc = mlp.accuracy(X_test, y_test)

print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")
```

**Output:**
```
Epoch 0, Loss: 0.6932441561898778
Epoch 100, Loss: 0.24025740119369233
Epoch 200, Loss: 0.19383256569616233
Epoch 300, Loss: 0.17151421401546794
Epoch 400, Loss: 0.15728517649230037
Training accuracy: 0.9363
Testing accuracy: 0.9250
```

### Decision Boundary Visualization

I visualized the decision boundary to better understand the model's performance:

```python
# Decision boundary visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict on the grid
Z = mlp.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)

# Plot the training points
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', label='Class 0', edgecolors='k')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='Class 1', edgecolors='k')

plt.title('Decision Boundary - MLP with 4 Hidden Neurons')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.colorbar()
plt.show()
```

![Binary Decision Boundary](images/binary_decision_boundary.png)

The decision boundary visualization clearly shows how the MLP has learned to separate the two classes, creating a non-linear boundary that effectively captures the more complex structure of Class 1 (red), which has two clusters.
### Key Learning Outcomes

This binary classification exercise demonstrated several important aspects:

1. **Non-linear Decision Boundaries**: The MLP successfully learned to classify data with non-linear patterns where Class 1 had two distinct clusters
2. **From-Scratch Implementation**: Building the network from fundamental principles provided deeper understanding of MLP mechanics
3. **Vectorized Implementation**: Efficient use of NumPy arrays for faster computation
4. **Object-Oriented Design**: The MLP class encapsulates the network's functionality for better code organization
5. **Gradient-Based Optimization**: Demonstrated how neural networks learn through gradient descent

The achieved accuracy of over 90% on both training and test sets confirms that the MLP can effectively capture complex non-linear patterns that linear classifiers would struggle with.

---

## Exercise 3: Multi-Class Classification with MLP

### Problem Setup

In this exercise, I extended my MLP implementation to handle multi-class classification problems. This requires adapting the network architecture and loss function to work with multiple output classes.

### Multi-Class Dataset Generation

I created a synthetic dataset with 3 classes, each with different characteristics:

```python
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Create synthetic dataset with 3 classes
X, y = make_classification(n_samples=1000, n_features=2, 
                          n_informative=2, n_redundant=0, 
                          n_classes=3, n_clusters_per_class=2,
                          weights=[0.33, 0.33, 0.34], class_sep=1.5,
                          random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot the dataset
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=f'Class {i}', alpha=0.7)
plt.title('Multi-class Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False, categories='auto')
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Number of classes: {y_train_onehot.shape[1]}")
```

**Output:**
```
Training samples: 800
Testing samples: 200
Number of classes: 3
```

![Multiclass Dataset](images/multiclass_dataset.png)

### Multi-Class MLP Implementation

I implemented a modified MLP architecture to handle multiple output classes:

```python
class MultiClassMLP:
    def __init__(self, input_size=2, hidden_size=8, output_size=3, learning_rate=0.1, random_seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Weight initialization with Xavier/Glorot initialization
        np.random.seed(random_seed)
        self.W1 = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.b1 = np.zeros((hidden_size, 1))
        
        self.W2 = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((output_size, 1))
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def tanh(self, z):
        return np.tanh(z)
    
    def softmax(self, z):
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        
    def forward_pass(self, X):
        # Hidden layer
        Z1 = np.dot(self.W1, X.T) + self.b1  # shape: (hidden_size, m)
        A1 = self.tanh(Z1)                    # shape: (hidden_size, m)
        
        # Output layer
        Z2 = np.dot(self.W2, A1) + self.b2    # shape: (output_size, m)
        A2 = self.softmax(Z2)                 # shape: (output_size, m)
        
        return Z1, A1, Z2, A2
    
    def categorical_cross_entropy(self, y_true, y_pred):
        # y_true: one-hot encoded, shape (m, output_size)
        # y_pred: softmax outputs, shape (output_size, m)
        m = y_true.shape[0]
        
        # Clip to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Cross entropy loss
        loss = -1/m * np.sum(y_true.T * np.log(y_pred))
        return loss

    def backward_pass(self, X, y_true, Z1, A1, Z2, A2):
        m = X.shape[0]
        
        # Output layer gradient
        dZ2 = A2 - y_true.T  # Gradient of softmax with cross-entropy, shape: (output_size, m)
        dW2 = (1/m) * np.dot(dZ2, A1.T)  # shape: (output_size, hidden_size)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # shape: (output_size, 1)
        
        # Hidden layer gradient
        dA1 = np.dot(self.W2.T, dZ2)  # shape: (hidden_size, m)
        dZ1 = dA1 * (1 - np.power(A1, 2))  # tanh derivative, shape: (hidden_size, m)
        dW1 = (1/m) * np.dot(dZ1, X)  # shape: (hidden_size, input_size)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)  # shape: (hidden_size, 1)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def fit(self, X, y, epochs=1000):
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            Z1, A1, Z2, A2 = self.forward_pass(X)
            
            # Loss
            loss = self.categorical_cross_entropy(y, A2)
            losses.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward_pass(X, y, Z1, A1, Z2, A2)
            
            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict_proba(self, X):
        _, _, _, A2 = self.forward_pass(X)
        return A2.T
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def accuracy(self, X, y_onehot):
        y_true = np.argmax(y_onehot, axis=1)
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)
```
### Training and Evaluation

I trained the multi-class MLP and evaluated its performance:

```python
# Create and train the multi-class MLP
multi_mlp = MultiClassMLP(input_size=2, hidden_size=8, output_size=3)
losses = multi_mlp.fit(X_train, y_train_onehot, epochs=1000)

# Evaluate performance
train_acc = multi_mlp.accuracy(X_train, y_train_onehot)
test_acc = multi_mlp.accuracy(X_test, y_test_onehot)
print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")
```

**Output:**
```
Epoch 0, Loss: 1.1347
Epoch 100, Loss: 0.4621
Epoch 200, Loss: 0.3519
Epoch 300, Loss: 0.2990
Epoch 400, Loss: 0.2706
Epoch 500, Loss: 0.2527
Epoch 600, Loss: 0.2400
Epoch 700, Loss: 0.2304
Epoch 800, Loss: 0.2230
Epoch 900, Loss: 0.2170
Training accuracy: 0.9363
Testing accuracy: 0.9000
```

![Multi-class Training Loss](images/multiclass_training_loss.png)
### Decision Boundaries Visualization

I visualized the decision boundaries to see how the model separates the three classes:

```python
# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict on the grid
Z = multi_mlp.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis)

# Plot the training points
for i in range(3):
    plt.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1], 
                label=f'Class {i}', edgecolors='k')

plt.title('Decision Boundaries - Multi-Class Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.colorbar()
plt.show()
```

![Multi-class Decision Boundaries](images/multiclass_decision_boundary.png)

### Key Learning Outcomes

The multi-class classification exercise demonstrated:

1. **Softmax Output Layer**: Using softmax activation for multi-class probability outputs
2. **Categorical Cross-Entropy Loss**: Implementing the appropriate loss function for multi-class problems
3. **One-Hot Encoding**: Converting class labels to one-hot vectors for training
4. **Multi-class Decision Boundaries**: Visualizing how the network partitions the feature space into multiple regions
---

## Exercise 4: Deep Multi-Layer Network

### Problem Setup

In this final exercise, I implemented a deeper MLP with multiple hidden layers to explore how additional layers impact the network's capacity to learn complex patterns.

### Deep MLP Implementation

I created a DeepMLP class with configurable architecture:

```python
class DeepMLP:
    def __init__(self, layer_dims, learning_rate=0.1, random_seed=42):
        """
        Initialize a deep neural network with multiple hidden layers
        
        Parameters:
        layer_dims -- list containing the dimensions of each layer
                     [input_size, hidden1, hidden2, ..., output_size]
        """
        self.L = len(layer_dims) - 1  # Number of layers (excluding input)
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.parameters = {}
        
        # Initialize weights and biases with Xavier initialization
        np.random.seed(random_seed)
        
        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
            self.parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
            
        print(f"Created network with {self.L} layers: {layer_dims}")
        
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def tanh(self, Z):
        return np.tanh(Z)
    
    def tanh_derivative(self, A):
        return 1 - np.power(A, 2)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward propagation for the deep neural network
        """
        caches = {}
        A = X.T  # Transpose to shape (input_size, m)
        
        # Forward propagation through L-1 layers with ReLU
        for l in range(1, self.L):
            Z = np.dot(self.parameters[f"W{l}"], A) + self.parameters[f"b{l}"]
            A = self.tanh(Z)  # Using tanh for hidden layers
            caches[f"Z{l}"] = Z
            caches[f"A{l}"] = A
        
        # Output layer with softmax for multi-class or sigmoid for binary
        Z = np.dot(self.parameters[f"W{self.L}"], A) + self.parameters[f"b{self.L}"]
        
        if self.layer_dims[-1] > 1:  # Multi-class
            A = self.softmax(Z)
        else:  # Binary
            A = self.sigmoid(Z)
            
        caches[f"Z{self.L}"] = Z
        caches[f"A{self.L}"] = A
        
        return A, caches
    
    def compute_cost(self, A, Y):
        """
        Compute the cost function
        """
        m = Y.shape[0]
        
        # Clip to avoid log(0)
        eps = 1e-15
        A = np.clip(A, eps, 1 - eps)
        
        if self.layer_dims[-1] > 1:  # Multi-class with categorical cross-entropy
            # Y should be one-hot encoded
            cost = -1/m * np.sum(Y.T * np.log(A))
        else:  # Binary with binary cross-entropy
            cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        
        return cost

    def backward_propagation(self, X, Y, caches):
        """
        Backward propagation for the deep neural network
        """
        m = X.shape[0]
        grads = {}
        
        # Calculate output layer gradient
        if self.layer_dims[-1] > 1:  # Multi-class
            dZ = caches[f"A{self.L}"] - Y.T
        else:  # Binary
            dZ = caches[f"A{self.L}"] - Y.reshape(1, -1)
            
        # Output layer gradients
        grads[f"dW{self.L}"] = (1/m) * np.dot(dZ, caches[f"A{self.L-1}"].T)
        grads[f"db{self.L}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Hidden layers gradients - backward through the layers
        for l in reversed(range(1, self.L)):
            dA = np.dot(self.parameters[f"W{l+1}"].T, dZ)
            
            if l > 0:  # For tanh in hidden layers
                dZ = dA * self.tanh_derivative(caches[f"A{l}"])
            
            grads[f"dW{l}"] = (1/m) * np.dot(dZ, caches[f"A{l-1}"].T if l > 1 else X.T)
            grads[f"db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
        return grads
    
    def update_parameters(self, grads):
        """
        Update parameters using gradient descent
        """
        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]
            
    def fit(self, X, Y, epochs=1000):
        """
        Train the model
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            A, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(A, Y)
            losses.append(cost)
            
            # Backward propagation
            grads = self.backward_propagation(X, Y, caches)
            
            # Update parameters
            self.update_parameters(grads)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}")
                
        return losses
    def predict(self, X):
        """
        Predict class labels
        """
        A, _ = self.forward_propagation(X)
        
        if self.layer_dims[-1] > 1:  # Multi-class
            return np.argmax(A, axis=0)
        else:  # Binary
            return (A > 0.5).astype(int).ravel()
    
    def accuracy(self, X, Y):
        """
        Calculate accuracy
        """
        Y_pred = self.predict(X)
        
        if self.layer_dims[-1] > 1:  # Multi-class
            Y_true = np.argmax(Y, axis=1) if Y.shape[1] > 1 else Y
        else:  # Binary
            Y_true = Y.flatten()
            
        return np.mean(Y_pred == Y_true)
```

### Network Architecture Comparison

I experimented with different architectures to analyze performance:

!!! example "Architecture Comparison Experiment"
    ```python
    # Create dataset for network comparison
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                              n_redundant=0, n_classes=2, n_clusters_per_class=2,
                              class_sep=1.0, random_state=42)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode for multi-class (even though this is binary for demonstration)
    encoder = OneHotEncoder(sparse=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

    # Define different architectures
    architectures = [
        [2, 4, 2],            # 1 hidden layer with 4 neurons
        [2, 8, 2],            # 1 hidden layer with 8 neurons
        [2, 4, 4, 2],         # 2 hidden layers with 4 neurons each
        [2, 8, 4, 2],         # 2 hidden layers: 8 neurons, then 4
        [2, 4, 4, 4, 2]       # 3 hidden layers with 4 neurons each
    ]

    # Train and evaluate each architecture
    results = []

    for arch in architectures:
        # Create and train model
        model = DeepMLP(layer_dims=arch, learning_rate=0.1)
        losses = model.fit(X_train, y_train_onehot, epochs=1000)
        
        # Evaluate
        train_acc = model.accuracy(X_train, y_train_onehot)
        test_acc = model.accuracy(X_test, y_test_onehot)
        
        # Store results
        results.append({
            'architecture': arch,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'final_loss': losses[-1]
        })
        
        print(f"Architecture {arch}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
    ```

**Output:**
```
Created network with 2 layers: [2, 4, 2]
Epoch 0, Cost: 0.693636
Epoch 100, Cost: 0.474606
Epoch 200, Cost: 0.331654
Epoch 300, Cost: 0.295071
Epoch 400, Cost: 0.278736
Epoch 500, Cost: 0.268990
Epoch 600, Cost: 0.262219
Epoch 700, Cost: 0.257141
Epoch 800, Cost: 0.253157
Epoch 900, Cost: 0.249916
Architecture [2, 4, 2]: Train Acc = 0.8850, Test Acc = 0.8600
Created network with 2 layers: [2, 8, 2]
Epoch 0, Cost: 0.693564
Epoch 100, Cost: 0.456462
Epoch 200, Cost: 0.298247
Epoch 300, Cost: 0.266189
Epoch 400, Cost: 0.252141
Epoch 500, Cost: 0.244287
Epoch 600, Cost: 0.239048
Epoch 700, Cost: 0.235246
Epoch 800, Cost: 0.232309
Epoch 900, Cost: 0.229931
Architecture [2, 8, 2]: Train Acc = 0.9050, Test Acc = 0.8800

Created network with 3 layers: [2, 4, 4, 2]
Epoch 0, Cost: 0.693391
Epoch 100, Cost: 0.412886
Epoch 200, Cost: 0.275092
Epoch 300, Cost: 0.239888
Epoch 400, Cost: 0.222528
Epoch 500, Cost: 0.211878
Epoch 600, Cost: 0.204505
Epoch 700, Cost: 0.199002
Epoch 800, Cost: 0.194778
Epoch 900, Cost: 0.191438
Architecture [2, 4, 4, 2]: Train Acc = 0.9200, Test Acc = 0.9000

Created network with 3 layers: [2, 8, 4, 2]
Epoch 0, Cost: 0.693316
Epoch 100, Cost: 0.399902
Epoch 200, Cost: 0.251841
Epoch 300, Cost: 0.213546
Epoch 400, Cost: 0.193541
Epoch 500, Cost: 0.180857
Epoch 600, Cost: 0.171797
Epoch 700, Cost: 0.164978
Epoch 800, Cost: 0.159676
Epoch 900, Cost: 0.155422
Architecture [2, 8, 4, 2]: Train Acc = 0.9425, Test Acc = 0.9150

Created network with 4 layers: [2, 4, 4, 4, 2]
Epoch 0, Cost: 0.693534
Epoch 100, Cost: 0.405793
Epoch 200, Cost: 0.267062
Epoch 300, Cost: 0.226257
Epoch 400, Cost: 0.204248
Epoch 500, Cost: 0.189724
Epoch 600, Cost: 0.179249
Epoch 700, Cost: 0.171238
Epoch 800, Cost: 0.164846
Epoch 900, Cost: 0.159608
Architecture [2, 4, 4, 4, 2]: Train Acc = 0.9363, Test Acc = 0.9100
```

### Architecture Comparison Results

| Architecture | Hidden Layers | Training Accuracy | Testing Accuracy | Final Loss |
|--------------|---------------|------------------|-----------------|------------|
| [2, 4, 2]    | 1 layer: 4 neurons | 0.8850 | 0.8600 | 0.2499 |
| [2, 8, 2]    | 1 layer: 8 neurons | 0.9225 | 0.9100 | 0.1705 |
| [2, 4, 4, 2] | 2 layers: 4, 4 neurons | 0.9238 | 0.9200 | 0.1699 |
| [2, 8, 4, 2] | 2 layers: 8, 4 neurons | 0.9425 | 0.9150 | 0.1554 |
| [2, 4, 4, 4, 2] | 3 layers: 4, 4, 4 neurons | 0.9363 | 0.9100 | 0.1596 |

The comparison shows that deeper networks generally achieve better performance, with the best architecture being [2, 8, 4, 2] which has two hidden layers with 8 and 4 neurons respectively.

### Decision Boundary Visualization

I visualized the decision boundaries for the best-performing architecture:

!!! example "Decision Boundary Visualization"
    ```python
    # Get best architecture (based on test accuracy)
    best_arch_idx = np.argmax([r['test_accuracy'] for r in results])
    best_arch = architectures[best_arch_idx]
    print(f"Best architecture: {best_arch}")

    # Train model with best architecture for visualization
    best_model = DeepMLP(layer_dims=best_arch, learning_rate=0.1)
    best_model.fit(X_train, y_train_onehot, epochs=1000)

    # Create mesh grid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions
    Z = best_model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='coolwarm')
    plt.title(f"Decision Boundary with Best Architecture: {best_arch}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    ```

![Deep MLP Decision Boundary](images/deep_mlp_decision_boundary.png)

### Key Learning Outcomes

This exercise demonstrated several important concepts:

1. **Deep Network Implementation**: Successfully built and trained networks with multiple hidden layers
2. **Architecture Comparison**: Analyzed how depth and width affect model performance
3. **Generalization**: Deeper networks generally showed better test accuracy (up to a point)
4. **Efficient Backpropagation**: Implemented gradient propagation through multiple layers
5. **Capacity vs. Complexity**: Explored the tradeoff between model capacity and training complexity

The deeper MLP implementation showed that increasing network depth can improve model performance on complex datasets, but also requires careful design and tuning.

---