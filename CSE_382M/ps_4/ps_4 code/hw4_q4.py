import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import KFold


def linear_regression_regularized(A, b, theta):
    r = A.shape[1]
    Z = A.T @ A + theta * np.eye(r)
    y = A.T @ b
    w = np.linalg.solve(Z, y)
    return w

df = pd.DataFrame(columns=["Dataset", "Best Theta", "Test Error"])

for i in range(2):

    # Load the in the dataset
    fl=np.load('CSE_382M/ps_4/ps_4 code/datasets/dataset{}.npz'. format(i+1))
    xtrain = fl['xtrain'].T
    ytrain = fl['ytrain'].T

    # regualarization parameter theta
    thetas = [1e-4,1e-3,1e-2,1e-1,1,10,100]

    # errors
    err = {}

    # Create K-Fold Cross-Validation split of the training dataset
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf.get_n_splits(xtrain)
    for train_index, test_index in kf.split(xtrain):
        x_train, x_test = xtrain[train_index], xtrain[test_index]
        y_train, y_test = ytrain[train_index], ytrain[test_index]

        # Loop over the regularization parameters
        for theta in thetas:
            feature_map = Nystroem(kernel='rbf', gamma=1, n_components=100)
            x_train_transformed = feature_map.fit_transform(x_train)
            x_test_transformed = feature_map.transform(x_test)
            
            # Train the model
            w = linear_regression_regularized(x_train_transformed, y_train, theta)
            
            # Make predictions
            y_pred = x_test_transformed @ w
            # Compute the error
            error = np.mean((y_test - y_pred) ** 2)
            # Store the error
            if theta not in err:
                err[theta] = []
            err[theta].append(error)
    # Compute the average error for each theta
    for theta in err:
        err[theta] = np.mean(err[theta])

    # pick best theta
    best_theta = min(err, key=err.get)

    # test the best theta
    yTest = fl['ytest'].T
    xTest = fl['xtest'].T
    xtrain = fl['xtrain'].T
    ytrain = fl['ytrain'].T
    
    feature_map = Nystroem(kernel='rbf', gamma=1, n_components=100)
    xtrain_transformed = feature_map.fit_transform(xtrain)
    xTest_transformed = feature_map.transform(xTest)
    w = linear_regression_regularized(xtrain_transformed, ytrain, best_theta)
    y_pred = xTest_transformed @ w

    # Compute the error
    error = np.mean((yTest - y_pred) ** 2)
    # Append the results to the DataFrame
    df = pd.concat([df, pd.DataFrame({"Dataset": [i+1], "Best Theta": [best_theta], "Test Error": [error]})], ignore_index=True)

df.to_csv("dataset_errors.csv", index=False)

print(df.to_latex(index=False,
                  float_format="{:.3E}".format,
))  