import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Set overall figure style and size
plt.rcParams['figure.figsize'] = (10, 8)
plt.style.use('seaborn-v0_8-whitegrid')

# Exercise 1 - Manual calculation MLP
def generate_exercise1_plots():
    # Setup the network parameters
    x = np.array([0.5, -0.2])
    y_true = 1

    W1 = np.array([[0.3, -0.1], 
                [0.2, 0.4]])
    b1 = np.array([0.1, -0.2])

    W2 = np.array([0.5, -0.3])
    b2 = 0.2

    learning_rate = 0.3

    def tanh(z):
        return np.tanh(z)

    def tanh_derivative(z):
        return 1 - np.tanh(z)**2
    
    # Forward Pass
    z1 = np.dot(W1, x) + b1
    a = tanh(z1)
    z2 = np.dot(W2, a) + b2
    y_pred = tanh(z2)
    
    # Loss Calculation
    N = 1
    L = (1 / N) * (y_true - y_pred) ** 2
    
    # Backward Pass
    dL_dy = 2 * (y_pred - y_true)
    dL_dz2 = dL_dy * tanh_derivative(z2)
    dL_dW2 = dL_dz2 * a
    dL_db2 = dL_dz2
    dL_da = W2 * dL_dz2
    dL_dz1 = dL_da * tanh_derivative(z1)
    dL_dW1 = np.outer(dL_dz1, x)
    dL_db1 = dL_dz1
    
    # Parameter Update
    W2_new = W2 - learning_rate * dL_dW2
    b2_new = b2 - learning_rate * dL_db2
    W1_new = W1 - learning_rate * dL_dW1
    b1_new = b1 - learning_rate * dL_db1
    
    # Create visualization of the network architecture
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define node positions
    input_pos = [(0, 0), (0, -2)]
    hidden_pos = [(2, 1), (2, -3)]
    output_pos = [(4, -1)]
    
    # Draw nodes
    ax.scatter(*zip(*input_pos), s=300, c='skyblue', label='Input Layer')
    ax.scatter(*zip(*hidden_pos), s=300, c='lightgreen', label='Hidden Layer')
    ax.scatter(*zip(*output_pos), s=300, c='salmon', label='Output Layer')
    
    # Add text to nodes
    for i, pos in enumerate(input_pos):
        ax.annotate(f'x{i+1}={x[i]}', pos, fontsize=12, ha='center', va='center')
    
    for i, pos in enumerate(hidden_pos):
        ax.annotate(f'h{i+1}={a[i]:.4f}', pos, fontsize=12, ha='center', va='center')
    
    ax.annotate(f'ŷ={y_pred:.4f}', output_pos[0], fontsize=12, ha='center', va='center')
    
    # Draw edges from input to hidden
    for i, i_pos in enumerate(input_pos):
        for j, h_pos in enumerate(hidden_pos):
            plt.plot([i_pos[0], h_pos[0]], [i_pos[1], h_pos[1]], 'k-', alpha=0.3)
            # Add weight label
            weight = W1[j, i]
            mid_x = (i_pos[0] + h_pos[0]) / 2
            mid_y = (i_pos[1] + h_pos[1]) / 2
            ax.annotate(f'{weight:.2f}', (mid_x, mid_y), fontsize=10)
    
    # Draw edges from hidden to output
    for i, h_pos in enumerate(hidden_pos):
        plt.plot([h_pos[0], output_pos[0][0]], [h_pos[1], output_pos[0][1]], 'k-', alpha=0.3)
        # Add weight label
        weight = W2[i]
        mid_x = (h_pos[0] + output_pos[0][0]) / 2
        mid_y = (h_pos[1] + output_pos[0][1]) / 2
        ax.annotate(f'{weight:.2f}', (mid_x, mid_y), fontsize=10)
    
    # Add bias labels
    ax.annotate(f'b1={b1[0]:.2f}', (2, 1.5), fontsize=10)
    ax.annotate(f'b1={b1[1]:.2f}', (2, -3.5), fontsize=10)
    ax.annotate(f'b2={b2:.2f}', (4, -0.5), fontsize=10)
    
    # Set plot properties
    ax.set_xlim(-1, 5)
    ax.set_ylim(-4, 2)
    ax.set_title('MLP Architecture for Manual Calculation', fontsize=14)
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/manual_calculation_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a parameter table visualization
    data = {
        "Param": ["W2[0]", "W2[1]", "b2", 
                  "W1[0,0]", "W1[0,1]", "W1[1,0]", "W1[1,1]", 
                  "b1[0]", "b1[1]"],
        
        "Initial": [W2[0], W2[1], b2, 
                    W1[0,0], W1[0,1], W1[1,0], W1[1,1], 
                    b1[0], b1[1]],
        
        "Updated": [W2_new[0], W2_new[1], b2_new, 
                   W1_new[0,0], W1_new[0,1], W1_new[1,0], W1_new[1,1], 
                   b1_new[0], b1_new[1]]
    }
    
    # Create parameter comparison figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(data)
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title('Parameter Updates After Gradient Descent', fontsize=14)
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/parameter_updates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Exercise 1 plots generated successfully!")

# Exercise 2 - Binary Classification
def generate_exercise2_plots():
    # Generate dataset
    N = 1000
    
    # Generate class 0 with 1 cluster
    X0, y0 = make_classification(n_samples=500, n_features=2, 
                                n_informative=2, n_redundant=0, 
                                n_clusters_per_class=1, n_classes=2, 
                                weights=[1.0, 0.0], # force all to one class
                                class_sep=1.5, random_state=42)

    # Generate class 1 with 2 clusters
    X1, y1 = make_classification(n_samples=500, n_features=2, 
                                n_informative=2, n_redundant=0, 
                                n_clusters_per_class=2, n_classes=2, 
                                weights=[0.0, 1.0], # force all to other class
                                class_sep=1.5, random_state=24)

    # Combine the datasets
    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.title("Binary Classification Dataset", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.colorbar(label='Class')
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/binary_classification_dataset.png', dpi=300)
    plt.close()
    
    # Define the MLP architecture for visualization
    input_size = 2
    hidden_size = 4
    output_size = 1
    
    # Initialize model for training
    np.random.seed(42)  # For reproducibility
    W1 = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
    b2 = np.zeros((output_size, 1))
    
    # Activation functions
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def tanh(z):
        return np.tanh(z)
    
    # Forward pass
    def forward_pass(X, W1, b1, W2, b2):
        # Hidden layer
        z1 = np.dot(W1, X.T) + b1  # Shape: (hidden_size, N)
        a1 = tanh(z1)              # Shape: (hidden_size, N)
        
        # Output layer
        z2 = np.dot(W2, a1) + b2   # Shape: (output_size, N)
        a2 = sigmoid(z2)           # Shape: (output_size, N)
        
        return z1, a1, z2, a2
    
    # Binary cross-entropy loss
    def binary_cross_entropy(y_true, y_pred):
        # To avoid log(0), we clip the predictions
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_pred = y_pred.flatten()
        m = y_true.shape[0]
        loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    # Backward pass
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
    
    # Parameter update
    def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        return W1, b1, W2, b2
    
    # Training loop
    epochs = 500
    learning_rate = 0.1
    losses = []
    
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
    
    # Plot training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='b')
    plt.title("Binary Classification - Training Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/binary_training_loss.png', dpi=300)
    plt.close()
    
    # Prediction function
    def predict(X, W1, b1, W2, b2):
        _, _, _, A2 = forward_pass(X, W1, b1, W2, b2)
        return (A2 >= 0.5).astype(int)
    
    # Decision boundary visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(grid_points, W1, b1, W2, b2)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm, s=20)
    plt.title("Binary Classification - Decision Boundary", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/binary_decision_boundary.png', dpi=300)
    plt.close()
    
    print("Exercise 2 plots generated successfully!")

# Exercise 3 - Multi-class Classification
def generate_exercise3_plots():
    # Generate dataset
    N = 1500
    
    # Generate class 0 with 2 clusters
    X0, y0 = make_classification(n_samples=N//3, n_features=4, 
                                n_informative=4, n_redundant=0, 
                                n_clusters_per_class=2, n_classes=3, 
                                weights=[1.0, 0.0, 0.0], # force all to one class
                                class_sep=1.5, random_state=42)

    # Generate class 1 with 3 clusters
    X1, y1 = make_classification(n_samples=N//3, n_features=4, 
                                n_informative=4, n_redundant=0, 
                                n_clusters_per_class=3, n_classes=3, 
                                weights=[0.0, 1.0, 0.0], # force all to other class
                                class_sep=1.5, random_state=24)

    # Generate class 2 with 4 clusters
    X2, y2 = make_classification(n_samples=N//3, n_features=4, 
                                n_informative=4, n_redundant=0, 
                                n_clusters_per_class=4, n_classes=3, 
                                weights=[0.0, 0.0, 1.0], # force all to third class
                                class_sep=1.5, random_state=100)

    # Combine the datasets
    X = np.vstack((X0, X1, X2))
    y = np.concatenate((y0, y1, y2))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Plot first 2 dimensions of the dataset
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', s=30, edgecolors='k')
    plt.title("Multi-class Dataset (First 2 Features)", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.colorbar(scatter, label='Class')
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/multiclass_dataset.png', dpi=300)
    plt.close()
    
    # Initialize ReusableMLP for training
    class ReusableMLP:
        def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, random_seed=42):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.learning_rate = learning_rate
            
            # Weights and biases
            np.random.seed(random_seed)
            self.W1 = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
            self.b1 = np.zeros((hidden_size, 1))
            
            self.W2 = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
            self.b2 = np.zeros((output_size, 1))
        
        def tanh(self, z):
            return np.tanh(z)
        
        def tanh_derivative(self, z):
            return 1 - np.tanh(z)**2
        
        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
        
        def softmax(self, z):
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Stability improvement
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        
        def forward_pass(self, X):
            # Hidden layer
            z1 = np.dot(self.W1, X.T) + self.b1  # Shape: (hidden_size, N)
            a1 = self.tanh(z1)                    # Shape: (hidden_size, N)
            
            # Output layer
            z2 = np.dot(self.W2, a1) + self.b2   # Shape: (output_size, N)
            if self.output_size > 1:
                a2 = self.softmax(z2)             # For multi-class classification
            else:
                a2 = self.sigmoid(z2)             # Shape: (output_size, N)
            
            return z1, a1, z2, a2
        
        def binary_cross_entropy(self, y_true, y_pred):
            # To avoid log(0), we clip the predictions
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            y_pred = y_pred.flatten()  # Ensure shape (N,)
            m = y_true.shape[0]
            loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            return loss
        
        def categorical_cross_entropy(self, y_true, y_pred):
            # To avoid log(0), we clip the predictions
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            m = y_true.shape[0]
            # Convert y_true to one-hot encoding
            y_one_hot = np.zeros((self.output_size, m))
            y_one_hot[y_true, np.arange(m)] = 1
            loss = - (1/m) * np.sum(y_one_hot * np.log(y_pred))
            return loss
        
        def backward_pass(self, X, y_true, Z1, A1, Z2, A2):
            m = X.shape[0]
            
            if self.output_size > 1:
                # Multi-class classification - different gradient calculation
                # Convert y_true to one-hot encoding for proper gradient calculation
                y_one_hot = np.zeros((self.output_size, m))
                y_one_hot[y_true, np.arange(m)] = 1
                dZ2 = A2 - y_one_hot                      # shape (output_size, m)
            else:
                # Binary classification
                dZ2 = A2 - y_true.reshape(1, -1)          # shape (1, m)
            
            dW2 = (1/m) * np.dot(dZ2, A1.T)           # shape (output_size, hidden_size)
            db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # shape (output_size, 1)
            
            # Hidden layer
            dA1 = np.dot(self.W2.T, dZ2)              # shape (hidden_size, m)
            dZ1 = dA1 * self.tanh_derivative(A1)      # tanh derivative applied to A1
            dW1 = (1/m) * np.dot(dZ1, X)              # shape (hidden_size, input_size)
            db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
            
            return dW1, db1, dW2, db2
        
        def update_parameters(self, dW1, db1, dW2, db2):
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        
        def predict(self, X):
            _, _, _, A2 = self.forward_pass(X)
            if self.output_size > 1:
                # Multi-class: return the class with highest probability
                return np.argmax(A2, axis=0)
            else:
                # Binary classification: threshold 0.5
                return (A2 >= 0.5).astype(int).flatten()
        
        def accuracy(self, X, y):
            y_pred = self.predict(X)
            return np.mean(y_pred == y)
        
        def fit(self, X, y, epochs=500):
            losses = []
            
            for epoch in range(epochs):
                # Forward pass
                Z1, A1, Z2, A2 = self.forward_pass(X)
                
                # Loss
                # Compute, but now with conditional
                if self.output_size > 1:
                    # For multi-class, we would use categorical cross-entropy
                    loss = self.categorical_cross_entropy(y, A2)
                else:
                    loss = self.binary_cross_entropy(y, A2)
                
                losses.append(loss)
                
                # Backward pass
                dW1, db1, dW2, db2 = self.backward_pass(X, y, Z1, A1, Z2, A2)
                
                # Update parameters
                self.update_parameters(dW1, db1, dW2, db2)
            
            return losses
    
    # Train the model
    model = ReusableMLP(input_size=4, hidden_size=16, output_size=3, learning_rate=0.05)
    losses = model.fit(X_train, y_train, epochs=500)
    
    # Plot the training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='g')
    plt.title("Multi-class Classification - Training Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Categorical Cross-Entropy Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/multiclass_training_loss.png', dpi=300)
    plt.close()
    
    # Decision boundary visualization (first 2 features)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # Create grid points with 4 features (padding the last 2 with mean values)
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_4d = np.column_stack([grid_2d, 
                            np.full(grid_2d.shape[0], X_train[:, 2].mean()),
                            np.full(grid_2d.shape[0], X_train[:, 3].mean())])
    
    # Get predictions using the trained model
    Z = model.predict(grid_4d)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis', levels=np.arange(4)-0.5)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='viridis', s=20)
    plt.title("Multi-class Classification - Decision Boundary", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.colorbar(ticks=[0, 1, 2], label='Class')
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/multiclass_decision_boundary.png', dpi=300)
    plt.close()
    
    print("Exercise 3 plots generated successfully!")

# Exercise 4 - Multi-class Classification with Deeper MLP
def generate_exercise4_plots():
    # Generate dataset
    N = 1500
    
    # Generate class 0 with 2 clusters
    X0, y0 = make_classification(n_samples=N//3, n_features=4, 
                                n_informative=4, n_redundant=0, 
                                n_clusters_per_class=2, n_classes=3, 
                                weights=[1.0, 0.0, 0.0], # force all to one class
                                class_sep=1.5, random_state=42)

    # Generate class 1 with 3 clusters
    X1, y1 = make_classification(n_samples=N//3, n_features=4, 
                                n_informative=4, n_redundant=0, 
                                n_clusters_per_class=3, n_classes=3, 
                                weights=[0.0, 1.0, 0.0], # force all to other class
                                class_sep=1.5, random_state=24)

    # Generate class 2 with 4 clusters
    X2, y2 = make_classification(n_samples=N//3, n_features=4, 
                                n_informative=4, n_redundant=0, 
                                n_clusters_per_class=4, n_classes=3, 
                                weights=[0.0, 0.0, 1.0], # force all to third class
                                class_sep=1.5, random_state=100)

    # Combine the datasets
    X = np.vstack((X0, X1, X2))
    y = np.concatenate((y0, y1, y2))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Deep MLP implementation
    class MLPDeep:
        def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr=0.1, seed=42):
            self.input_size = input_size
            self.hidden1_size = hidden1_size
            self.hidden2_size = hidden2_size
            self.output_size = output_size
            self.lr = lr
            
            np.random.seed(seed)
            # Xavier initialization
            self.W1 = np.random.randn(hidden1_size, input_size) / np.sqrt(input_size)
            self.b1 = np.zeros((hidden1_size, 1))
            
            self.W2 = np.random.randn(hidden2_size, hidden1_size) / np.sqrt(hidden1_size)
            self.b2 = np.zeros((hidden2_size, 1))
            
            self.W3 = np.random.randn(output_size, hidden2_size) / np.sqrt(hidden2_size)
            self.b3 = np.zeros((output_size, 1))

        def _tanh(self, z):
            return np.tanh(z)

        def _tanh_deriv(self, a):
            return 1 - np.power(a, 2)

        def _sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

        def _softmax(self, z):
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # numerical stability
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)

        def forward(self, X):
            # First hidden layer
            self.Z1 = np.dot(self.W1, X.T) + self.b1
            self.A1 = self._tanh(self.Z1)
            
            # Second hidden layer
            self.Z2 = np.dot(self.W2, self.A1) + self.b2
            self.A2 = self._tanh(self.Z2)
            
            # Output layer
            self.Z3 = np.dot(self.W3, self.A2) + self.b3
            if self.output_size > 1:
                self.A3 = self._softmax(self.Z3)  # Multi-class classification
            else:
                self.A3 = self._sigmoid(self.Z3)  # Binary classification
            
            return self.A3

        def binary_cross_entropy(self, y_true, y_pred):
            # To avoid log(0), we clip the predictions
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            y_pred = y_pred.flatten()  # Ensure shape (N,)
            m = y_true.shape[0]
            loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            return loss

        def categorical_cross_entropy(self, y_true, y_pred):
            # To avoid log(0), we clip the predictions
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            m = y_true.shape[0]
            # Convert y_true to one-hot encoding
            y_one_hot = np.zeros((self.output_size, m))
            y_one_hot[y_true, np.arange(m)] = 1
            loss = - (1/m) * np.sum(y_one_hot * np.log(y_pred))
            return loss

        def compute_loss(self, y_true, y_pred):
            if self.output_size > 1:
                return self.categorical_cross_entropy(y_true, y_pred)
            else:
                return self.binary_cross_entropy(y_true, y_pred)

        def backward(self, X, y_true):
            m = X.shape[0]

            if self.output_size > 1:
                # Multi-class classification - convert y_true to one-hot encoding
                y_one_hot = np.zeros((self.output_size, m))
                y_one_hot[y_true, np.arange(m)] = 1
                dZ3 = self.A3 - y_one_hot
            else:
                # Binary classification
                dZ3 = self.A3 - y_true.reshape(1, -1)

            # Output layer gradients
            dW3 = (1/m) * np.dot(dZ3, self.A2.T)
            db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

            # Hidden layer 2 gradients
            dA2 = np.dot(self.W3.T, dZ3)
            dZ2 = dA2 * self._tanh_deriv(self.A2)
            dW2 = (1/m) * np.dot(dZ2, self.A1.T)
            db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

            # Hidden layer 1 gradients
            dA1 = np.dot(self.W2.T, dZ2)
            dZ1 = dA1 * self._tanh_deriv(self.A1)
            dW1 = (1/m) * np.dot(dZ1, X)
            db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

            # Update parameters
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3

        def predict(self, X):
            probs = self.forward(X)
            if self.output_size > 1:
                # Multi-class: return the class with highest probability
                return np.argmax(probs, axis=0)
            else:
                # Binary classification: threshold 0.5
                return (probs >= 0.5).astype(int).flatten()

        def accuracy(self, X, y):
            y_pred = self.predict(X)
            return np.mean(y_pred == y)

        def fit(self, X, y, epochs=500):
            losses = []
            
            for epoch in range(epochs):
                # Forward pass
                y_pred = self.forward(X)
                
                # Compute loss
                loss = self.compute_loss(y, y_pred)
                losses.append(loss)
                
                # Backward pass
                self.backward(X, y)
            
            return losses
    
    # Create visual comparison of architectures
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Network 1 (Shallow) - left side
    ax.text(0.25, 0.95, "Shallow MLP Architecture", ha='center', fontsize=14, fontweight='bold')
    
    # Draw nodes
    # Input layer
    for i in range(4):
        ax.scatter(0.1, 0.8 - i*0.15, s=300, c='skyblue')
        ax.text(0.1, 0.8 - i*0.15, f'x{i+1}', ha='center', va='center', fontsize=12)
    
    # Hidden layer
    for i in range(16):
        ax.scatter(0.25, 0.9 - i*0.05, s=200, c='lightgreen')
    ax.text(0.25, 0.95, "16 neurons", ha='center', va='center', fontsize=12)
    
    # Output layer
    for i in range(3):
        ax.scatter(0.4, 0.75 - i*0.1, s=300, c='salmon')
        ax.text(0.4, 0.75 - i*0.1, f'y{i+1}', ha='center', va='center', fontsize=12)
    
    # Draw connections
    for i in range(4):
        for j in range(16):
            ax.plot([0.1, 0.25], [0.8 - i*0.15, 0.9 - j*0.05], 'k-', alpha=0.05)
    
    for i in range(16):
        for j in range(3):
            ax.plot([0.25, 0.4], [0.9 - i*0.05, 0.75 - j*0.1], 'k-', alpha=0.05)
    
    # Network 2 (Deep) - right side
    ax.text(0.75, 0.95, "Deep MLP Architecture", ha='center', fontsize=14, fontweight='bold')
    
    # Draw nodes
    # Input layer
    for i in range(4):
        ax.scatter(0.6, 0.8 - i*0.15, s=300, c='skyblue')
        ax.text(0.6, 0.8 - i*0.15, f'x{i+1}', ha='center', va='center', fontsize=12)
    
    # Hidden layer 1
    for i in range(16):
        ax.scatter(0.7, 0.9 - i*0.05, s=200, c='lightgreen')
    ax.text(0.7, 0.95, "16 neurons", ha='center', va='center', fontsize=12)
    
    # Hidden layer 2
    for i in range(8):
        ax.scatter(0.8, 0.85 - i*0.08, s=200, c='lightgreen')
    ax.text(0.8, 0.9, "8 neurons", ha='center', va='center', fontsize=12)
    
    # Output layer
    for i in range(3):
        ax.scatter(0.9, 0.75 - i*0.1, s=300, c='salmon')
        ax.text(0.9, 0.75 - i*0.1, f'y{i+1}', ha='center', va='center', fontsize=12)
    
    # Draw connections
    for i in range(4):
        for j in range(16):
            ax.plot([0.6, 0.7], [0.8 - i*0.15, 0.9 - j*0.05], 'k-', alpha=0.05)
    
    for i in range(16):
        for j in range(8):
            ax.plot([0.7, 0.8], [0.9 - i*0.05, 0.85 - j*0.08], 'k-', alpha=0.05)
    
    for i in range(8):
        for j in range(3):
            ax.plot([0.8, 0.9], [0.85 - i*0.08, 0.75 - j*0.1], 'k-', alpha=0.05)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title("Architecture Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/architecture_comparison.png', dpi=300)
    plt.close()
    
    # Train the model
    model_deep = MLPDeep(input_size=4, hidden1_size=16, hidden2_size=8, output_size=3, lr=0.05)
    losses = model_deep.fit(X_train, y_train, epochs=500)
    
    # Plot the training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='purple')
    plt.title("Deep MLP - Training Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Categorical Cross-Entropy Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/deep_mlp_training_loss.png', dpi=300)
    plt.close()
    
    # Decision boundary visualization (first 2 features)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # Create grid points with 4 features (padding the last 2 with mean values)
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_4d = np.column_stack([grid_2d, 
                            np.full(grid_2d.shape[0], X_train[:, 2].mean()),
                            np.full(grid_2d.shape[0], X_train[:, 3].mean())])
    
    # Get predictions using the deep model
    Z_deep = model_deep.predict(grid_4d)
    Z_deep = Z_deep.reshape(xx.shape)
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_deep, alpha=0.8, cmap='viridis', levels=np.arange(4)-0.5)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap='viridis', s=20)
    plt.title("Deep MLP - Decision Boundary", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.colorbar(ticks=[0, 1, 2], label='Class')
    plt.tight_layout()
    plt.savefig('docs/exercises/mlp/images/deep_mlp_decision_boundary.png', dpi=300)
    plt.close()
    
    print("Exercise 4 plots generated successfully!")

# Run all plot generation functions
if __name__ == "__main__":
    # Create the images directory if it doesn't exist
    import os
    os.makedirs('docs/exercises/mlp/images', exist_ok=True)
    
    # Generate all plots
    generate_exercise1_plots()
    generate_exercise2_plots()
    generate_exercise3_plots()
    generate_exercise4_plots()
    
    print("All plots generated successfully!")