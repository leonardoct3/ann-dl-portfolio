"""
Generate plots for the Perceptron Exercise documentation.
This script creates all the necessary visualizations mentioned in the exercise.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory for images
output_dir = "docs/exercises/perceptron/images"
os.makedirs(output_dir, exist_ok=True)

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("Set1")

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.accuracies = []
    
    def fit(self, X, y):
        # Initialize weights and bias
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels to -1 and 1 for perceptron algorithm
        y_train = np.where(y == 0, -1, 1)
        
        # Training loop
        for epoch in range(self.n_iters):
            errors = 0
            
            for i in range(X.shape[0]):
                # Calculate prediction
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else -1
                
                # Update weights if misclassified
                if y_train[i] != y_pred:
                    self.weights += self.learning_rate * y_train[i] * X[i]
                    self.bias += self.learning_rate * y_train[i]
                    errors += 1
            
            # Calculate accuracy for this epoch
            accuracy = self.score(X, y)
            self.accuracies.append(accuracy)
            
            # Check for convergence
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

def generate_exercise1_plots():
    """Generate plots for Exercise 1 - Linearly Separable Data"""
    print("\nGenerating Exercise 1 plots...")
    
    # Set seed for reproducibility in Exercise 1
    np.random.seed(42)
    
    # Generate Exercise 1 dataset
    sample = pd.DataFrame()
    
    # Class 0
    mu = [1.5, 1.5]
    cov = [[0.5, 0], [0, 0.5]]
    class_0 = np.random.multivariate_normal(mu, cov, 1000)
    sample['feature1'] = class_0[:, 0]
    sample['feature2'] = class_0[:, 1]
    sample['label'] = 0
    
    # Class 1
    mu = [5, 5]
    cov = [[0.5, 0], [0, 0.5]]
    class_1 = np.random.multivariate_normal(mu, cov, 1000)
    temp = pd.DataFrame()
    temp['feature1'] = class_1[:, 0]
    temp['feature2'] = class_1[:, 1]
    temp['label'] = 1
    sample = pd.concat([sample, temp], ignore_index=True)
    
    # Train perceptron
    X = sample[['feature1', 'feature2']].values
    y = sample['label'].values
    perceptron = Perceptron(learning_rate=0.01, n_iters=100)
    perceptron.fit(X, y)
    
    # Print results for documentation
    print("\nExercise 1 Results (for documentation):")
    print(f"Final Accuracy: {perceptron.score(X, y)*100:.2f}%")
    print(f"Final Weights: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}]")
    print(f"Final Bias: {perceptron.bias:.4f}")
    print(f"Convergence: Achieved at epoch {len(perceptron.accuracies)}")
    
    # Plot 1: Dataset visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=sample, x='feature1', y='feature2', hue='label', 
                   palette='Set1', alpha=0.7, s=50)
    plt.title('Exercise 1: Linearly Separable Classes Dataset', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/exercise1_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(perceptron.accuracies) + 1), perceptron.accuracies, 
             marker='o', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Training Accuracy Progression', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(f'{output_dir}/exercise1_training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Decision boundary
    plt.figure(figsize=(12, 8))
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    sns.scatterplot(data=sample, x='feature1', y='feature2', hue='label', 
                   palette='Set1', edgecolor='k', linewidth=0.5, s=50)
    plt.title('Perceptron Decision Boundary', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/exercise1_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return sample, perceptron

def generate_exercise2_plots():
    """Generate plots for Exercise 2 - Overlapping Classes"""
    print("\nGenerating Exercise 2 plots...")
    
    # Reset random seed to ensure randomness in Exercise 2
    np.random.seed(None)
    
    best_accuracy = 0
    best_perceptron = None
    best_sample = None
    results = []
    
    # Run multiple times to get consistent results
    for run in range(5):
        sample = pd.DataFrame()
        
        # Class 0
        mu = [3, 3]
        cov = [[1.5, 0], [0, 1.5]]
        class_0 = np.random.multivariate_normal(mu, cov, 1000)
        sample['feature1'] = class_0[:, 0]
        sample['feature2'] = class_0[:, 1]
        sample['label'] = 0
        
        # Class 1
        mu = [4, 4]
        cov = [[1.5, 0], [0, 1.5]]
        class_1 = np.random.multivariate_normal(mu, cov, 1000)
        temp = pd.DataFrame()
        temp['feature1'] = class_1[:, 0]
        temp['feature2'] = class_1[:, 1]
        temp['label'] = 1
        sample = pd.concat([sample, temp], ignore_index=True)
        
        # Train perceptron
        X = sample[['feature1', 'feature2']].values
        y = sample['label'].values
        perceptron = Perceptron(learning_rate=0.01, n_iters=100)
        perceptron.fit(X, y)
        
        accuracy = perceptron.score(X, y)
        results.append({
            'accuracy': accuracy,
            'epochs': len(perceptron.accuracies),
            'weights': perceptron.weights,
            'bias': perceptron.bias
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_perceptron = perceptron
            best_sample = sample
    
    # Print results for documentation
    print("\nExercise 2 Results (for documentation):")
    print("Multiple Runs Analysis:")
    for i, result in enumerate(results, 1):
        print(f"Run {i}: Accuracy = {result['accuracy']*100:.2f}%, Epochs = {result['epochs']}")
    
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    std_accuracy = np.std([r['accuracy'] for r in results])
    print(f"\nAverage accuracy: {avg_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    
    # Use best run for plots
    X = best_sample[['feature1', 'feature2']].values
    y = best_sample['label'].values
    
    # Plot 1: Dataset visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=best_sample, x='feature1', y='feature2', hue='label', 
                   palette='Set1', alpha=0.7, s=50)
    plt.title('Exercise 2: Overlapping Classes Dataset', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/exercise2_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_perceptron.accuracies) + 1), best_perceptron.accuracies, 
             marker='o', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Training Accuracy over Epochs', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(f'{output_dir}/exercise2_training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Decision boundary with misclassified points
    plt.figure(figsize=(12, 8))
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = best_perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    predictions = best_perceptron.predict(X)
    misclassified_mask = predictions != y
    correctly_classified_mask = predictions == y
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    plt.scatter(X[correctly_classified_mask, 0], X[correctly_classified_mask, 1], 
               c=y[correctly_classified_mask], cmap='Set1', alpha=0.7, s=50, 
               label='Correctly Classified', edgecolors='black', linewidth=0.5)
    plt.scatter(X[misclassified_mask, 0], X[misclassified_mask, 1], 
               c=y[misclassified_mask], cmap='Set1', alpha=0.9, s=100, 
               marker='x', linewidth=3, label='Misclassified')
    
    plt.title('Decision Boundary with Misclassified Points', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/exercise2_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_sample, best_perceptron

def generate_comparison_plots(sample1, perceptron1, sample2, perceptron2):
    """Generate comparison plots between the two exercises"""
    print("\nGenerating comparison plots...")
    
    # Dataset comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(data=sample1, x='feature1', y='feature2', hue='label', 
                   palette='Set1', alpha=0.7, s=50, ax=ax1)
    ax1.set_title('Exercise 1: Linearly Separable Classes', fontsize=14)
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    sns.scatterplot(data=sample2, x='feature1', y='feature2', hue='label', 
                   palette='Set1', alpha=0.7, s=50, ax=ax2)
    ax2.set_title('Exercise 2: Overlapping Classes', fontsize=14)
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(perceptron1.accuracies) + 1), perceptron1.accuracies, 
             marker='o', linewidth=2, markersize=6, label='Exercise 1 (Separable)')
    plt.plot(range(1, len(perceptron2.accuracies) + 1), perceptron2.accuracies, 
             marker='s', linewidth=2, markersize=4, label='Exercise 2 (Overlapping)')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Training Accuracy Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(f'{output_dir}/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all plots"""
    print("Starting plot generation for Perceptron Exercise...")
    
    # Generate Exercise 1 plots (with fixed seed)
    sample1, perceptron1 = generate_exercise1_plots()
    
    # Generate Exercise 2 plots (with multiple random runs)
    sample2, perceptron2 = generate_exercise2_plots()
    
    # Generate comparison plots
    generate_comparison_plots(sample1, perceptron1, sample2, perceptron2)
    
    print("\nAll plots generated successfully!")
    print("\nGenerated files:")
    for filename in os.listdir(output_dir):
        if filename.endswith('.png'):
            print(f"  - {filename}")

if __name__ == "__main__":
    main()
