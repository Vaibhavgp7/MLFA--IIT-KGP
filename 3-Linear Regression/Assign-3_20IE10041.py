# Vaibhav Gupta
# 20IE10041

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""---------------Experiment-1------------------"""

# load the dataset
df = pd.read_csv('BostonHousingDataset.csv')
df = df.drop(columns=['B','LSTAT'])
dataset_altered = df.dropna()
dataset_altered = dataset_altered.astype(float)
dataset_altered.head(10)

"""---------------Experiment-2------------------"""


# Plot histograms of 'NOX', 'RM', and 'AGE'
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(dataset_altered['NOX'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('NOX')
plt.ylabel('Frequency')
plt.title('Histogram of NOX')

plt.subplot(1, 3, 2)
plt.hist(dataset_altered['RM'], bins=20, color='lightgreen', edgecolor='black')
plt.xlabel('RM')
plt.ylabel('Frequency')
plt.title('Histogram of RM')

plt.subplot(1, 3, 3)
plt.hist(dataset_altered['AGE'], bins=20, color='salmon', edgecolor='black')
plt.xlabel('AGE')
plt.ylabel('Frequency')
plt.title('Histogram of AGE')

plt.tight_layout()
plt.show()

# Tabulate correlation coefficients for all columns
correlation_matrix = dataset_altered.corr()

print(correlation_matrix)

# Plot the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

"""---------------Experiment-3------------------"""

dataset_altered_features = dataset_altered.drop(columns=['MEDV'])  
dataset_altered_target = dataset_altered['MEDV']
print(dataset_altered_features.shape[0],dataset_altered_features.shape[1])
print(dataset_altered_target.shape[0])

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Standardize the features using Z-score normalization
scaler = StandardScaler()
dataset_altered_features = scaler.fit_transform(dataset_altered_features)
# print(dataset_altered_features)

X_train, X_test, y_train, y_test = train_test_split(dataset_altered_features,dataset_altered_target,test_size=0.1, random_state=100)

"""---------------Experiment-4------------------"""

#function for adding bias also for lr_closed form solution
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def lr_closed_form(X, y):
    # Add bias term to feature matrix
    X = add_bias(X)
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


# Perform LR_ClosedForm on the training data
theta = lr_closed_form(X_train, y_train)

print(theta)

# Add bias term to the testing feature matrix
X_test_bias = add_bias(X_test)
# Predict the target values
y_pred = X_test_bias.dot(theta)


# Calculate the Mean Squared Error (MSE) for evaluation
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("Root Mean Squared Error:", rmse)

"""---------------Experiment-5------------------"""

#class for Gradient Descent method
class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Performing gradient descent 
        for _ in range(self.num_iterations):
            
            predictions = self.predict(X)

            # Computing gradients
            dw = -(2 / len(X)) * np.dot(X.T, (y - predictions))
            db = -(2 / len(X)) * np.sum(y - predictions)

            # Updating weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Initialize lists to store RMSE values and optimal coefficients
rmse_vals = []
coeffs = []
learning_rates = [0.001, 0.01, 0.1]

for learning_rate in learning_rates:
    lr_model = LinearRegressionGradientDescent(learning_rate=learning_rate, num_iterations=1000)

    # Fit the model to the training data
    lr_model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = lr_model.predict(X_test)

    # Calculate Mean Squared Error (MSE) on test data
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Root Mean Squared Error for learning rate {learning_rate} =", rmse)

    rmse_vals.append(rmse)
    coeffs.append((lr_model.weights, lr_model.bias))

# Find the optimal learning rate
optimal_learning_rate = learning_rates[np.argmin(rmse_vals)]
optimal_index = np.argmin(rmse_vals)
optimal_rmse = rmse_vals[optimal_index]
optimal_weights = coeffs[optimal_index][0]
optimal_bias = coeffs[optimal_index][1]

# Print the optimal learning rate and corresponding RMSE
print("Optimal Learning Rate:", optimal_learning_rate)
print("Corresponding RMSE:", optimal_rmse)
print("Optimal Coefficients:", optimal_weights)
print("Optimal Bias:", optimal_bias)

# Plot RMSE vs Learning Rates
plt.bar(range(len(learning_rates)), rmse_vals, tick_label=[str(lr) for lr in learning_rates],color='purple')
plt.xlabel('Learning Rate')
plt.ylabel('RMSE')
plt.title('RMSE vs Learning Rates')
plt.show()




