import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegression:
    def __init__(self, Lambda, lr=0.01, num_iter=1000000, fit_intercept=True, verbose=False):
        self.pos_X, self.pos_y, self.neg_X, self.neg_y, self.labels = self.__read_data()
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.cost_history = []
        self.Lambda = Lambda

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        m = y.shape[0]
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + self.Lambda/(2 * m) * sum(self.theta ** 2)

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if self.verbose and i % 10000 == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                loss = self.__loss(h, y)
                self.cost_history.append(loss)
                print(f'loss: {loss} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

    # plotting positive and negative data points.
    def plot_data(self):
        plt.plot(self.pos_X, self.pos_y, 'ro')
        plt.plot(self.neg_X, self.neg_y, 'bo')
        plt.xlabel('Exam 1 score')
        plt.ylabel('Exam 2 score')
        plt.legend(['Not Admitted', 'Admitted'])
        plt.show()

    def plot_line(self, X, y):
        x_values = [np.min(self.neg_X), np.max(self.pos_X)]
        y_values = - (self.theta[0] + np.dot(self.theta[1], x_values)) / self.theta[2]
        plt.plot(self.pos_X, self.pos_y, 'ro')
        plt.plot(self.neg_X, self.neg_y, 'bo')
        # plt.plot(X, y, '-r')
        plt.plot(x_values, y_values, label='Decision Boundary')
        plt.xlabel('Marks in 1st Exam')
        plt.ylabel('Marks in 2nd Exam')
        plt.legend()
        plt.show()

    def __read_data(self):
        # Reads data from file and separates X , y into 2 classes
        # according to their label's.
        pos_X = []
        pos_y = []

        neg_X = []
        neg_y = []

        X = []
        labels = []
        with open('data.txt') as f:
            for line in f.readlines():
                x_value, y_value, label = line.replace('\n', '').split(',')
                X.append([float(x_value), float(y_value)])
                labels.append(float(label))
                if label == '1':
                    pos_X.append(float(x_value))
                    pos_y.append(float(y_value))
                elif label == '0':
                    neg_X.append(float(x_value))
                    neg_y.append(float(y_value))

        self.X = np.asarray(X)
        return np.asarray(pos_X), np.asarray(pos_y), np.asarray(neg_X), np.asarray(neg_y), labels

    def plot_cost(self):
        plt.plot(self.cost_history[10:], 'bo')
        plt.xlabel('#Iteration')
        plt.ylabel('Cost')
        plt.show()

    def compare_to_Sklearn(self, X, y):
        model = LR()
        model.fit(X, y)
        print("My LR Theta:", self.theta)
        print("Sklearn LR Theta:", model.coef_)
