from __future__ import print_function

from LogisticRegression import LogisticRegression
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.datasets import load_breast_cancer

import numpy as np

# loading Iris dataset containing 150 records.
X, y = load_breast_cancer(return_X_y=True)

# converting selected data to numpy arrays.
X = np.asarray(X[:, :2])
y = np.asarray(y)

# printing features denoted by O.
# plt.plot(X, '+')
# plt.show()

# declaring a list in size of selected dataset containing 0 to size of dataset.
indices = list(range(len(X)))

# shuffling indices array.
np.random.shuffle(indices)

num_training_samples = 500

# splitting train data arrays according to indices array.
x_train = X[indices[:num_training_samples]]
y_train = y[indices[:num_training_samples]]

# splitting test data arrays according to indices array.
x_test = X[indices[num_training_samples:]]
y_test = y[indices[num_training_samples:]]

# Defining sklearn.linear model with lbfgs solver
sk_model = skLR(C=100, solver='lbfgs')

# fitting sklearn.linear.LogisticRegression model over train set.
sk_model.fit(x_train, y_train)

# declaring my object of class logistic regression.
# LogisticRegression(X, y, num_iter=100, learning_rate=0.01, verbose=True)
model = LogisticRegression(X=x_train, y=y_train, num_iter=1000, learning_rate=0.01, Lambda=1, verbose=True)

# printing number of train set data points.
print()
print('Training %s numbers of train data points...' % num_training_samples)

# fitting model over train set.
model.train()

# printing Theta values of trained sklearn.linear model.
print('Sklearn coefficient:\n', sk_model.coef_)
print(sk_model.intercept_)

# printing Theta values of trained LogisticRegression model.
print('My coefficient:\n', model.theta)
print()

# plotting cost values over iterations.
model.plot_cost()

# predicting sklearn.linear model over test set.
sk_predict = sk_model.predict(x_test)

# predicting LogisticRegression model over test set.
my_predict = model.predict(x_test)

# printing original labels of test set.
print('True Labels:\n', y_test)

# printing sklearn.linear model prediction.
print('Sklearn.linear model predicts:\n', sk_predict)

# printing LogisticRegression model prediction.
print('LogisticRegression model predict:\n', my_predict)

# comparing prediction of sklearn.linear model and original test set labels.
# calculating accuracy:
accuracy = sum(sk_predict == y_test) / (X.shape[0] - num_training_samples)
print("Sklearn.linear model Accuracy: %.3f" % accuracy + " %")
print()

# comparing prediction of LogisticRegression model and original test set labels.
# calculating accuracy:
my_accuracy = sum(my_predict == y_test) / (X.shape[0] - num_training_samples)
print("LogisticRegression Accuracy: %.3f" % my_accuracy + " %")