#  Support Vector Classification for MNIST and Support vector regression for California Housing
# Name: Vaibhav Gupta
# Roll No: 20IE10041

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from tensorflow import keras
from keras.datasets import mnist


# SVC for MNIST dataset
def svc():
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten each input image into a vector of length 784
    X_train_flatten = X_train.reshape(X_train.shape[0], -1)
    X_test_flatten = X_test.reshape(X_test.shape[0], -1)

    # Normalize the pixel values of the images by dividing them by 255
    X_train_normalized = X_train_flatten / 255.0
    X_test_normalized = X_test_flatten / 255.0

    # Select the first 10,000 samples for training and the first 2,000 samples for testing
    X_train = X_train_normalized[:10000]
    y_train = y_train[:10000]
    X_test = X_test_normalized[:2000]
    y_test = y_test[:2000]
    

    # Train models with different kernels
    kernels = ['linear', 'poly', 'rbf']
    for kern in kernels:
        model = SVC(kernel=kern)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Classification report for kernel={kern}:\n{classification_report(y_test, y_pred)}")

    # Hyperparameter tuning with GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3)
    grid.fit(X_train, y_train)
    print("Best parameters from GridSearchCV:", grid.best_params_)

    # Train and evaluate model with best parameters
    best_model = SVC(kernel='rbf', **grid.best_params_)
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    #printing the classification report
    print("Classification report for best model:\n", classification_report(y_test, y_pred_best))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_best)
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for MNIST SVC')
    plt.show()

# SVR for California Housing dataset
def svr():
    # Load and split data
    hous = fetch_california_housing()
    X, y = hous.data, hous.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    # Train SVR with default parameters
    svr = SVR(epsilon=0.5)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    print("MSE with default parameter:", mean_squared_error(y_test, y_pred))

    # Scatter plot for default parameters
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('SVR Predictions vs Actual (Default Parameters)')
    plt.show()

    # Hyperparameter tuning with GridSearchCV
    eps = np.arange(0, 2.6, 0.1)
    param_grid = {'epsilon': eps}
    grid = GridSearchCV(SVR(), param_grid, cv=10)
    grid.fit(X_train, y_train)
    print("Best epsilon from GridSearchCV:", grid.best_params_['epsilon'])

    # Train and evaluate model with best epsilon
    best_svr = SVR(epsilon=grid.best_params_['epsilon'])
    best_svr.fit(X_train, y_train)
    y_best = best_svr.predict(X_test)
    print("MSE with best epsilon:", mean_squared_error(y_test, y_best))

    # Scatter plot for predictions with best epsilon
    plt.figure()
    plt.scatter(y_test, y_best, alpha=0.3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('SVR Predictions vs Actual (Best Epsilon)')
    plt.show()

# svc()
svr()
