from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression:
    def __init__(self, X, y, num_iter, learning_rate, Lambda, file_name='data.txt', verbose=False):
        self.pos_X, self.pos_y, self.neg_X, self.neg_y, self.labels = self.__read_data()
        self.y = np.asarray(self.labels)
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.cost_history = []
        self.X = X
        self.y = y
        self.theta = np.ones(self.X.shape[1] + 1)
        self.X = np.c_[np.ones([np.shape(self.X)[0], 1]), self.X]
        self.Lambda = Lambda
        self.file_path = file_name

    # Sigmoid function declaration that takes X and theta as input.
    def __sigmoid(self, X, theta):
        return 1 / (1 + np.exp((-np.dot(X, theta))))

    # plotting positive and negative data points.
    def plot_data(self):
        plt.plot(self.pos_X, self.pos_y, 'ro')
        plt.plot(self.neg_X, self.neg_y, 'bo')
        plt.xlabel('Exam 1 score')
        plt.ylabel('Exam 2 score')
        plt.legend(['Not Admitted', 'Admitted'])
        plt.show()

    def plot_line(self):
        x = np.linspace(min(np.append(self.neg_X, self.pos_X)), max(np.append(self.neg_X, self.pos_X)))
        y = self.theta[0] + self.theta[1] * x + self.theta[2] * x
        plt.plot(self.pos_X, self.pos_y, 'ro')
        plt.plot(self.neg_X, self.neg_y, 'bo')
        plt.plot(x, y, '-r')
        plt.xlabel('Data')
        plt.ylabel('Label')
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

    def __calculate_cost(self):
        m = self.y.shape[0]
        h = self.__sigmoid(self.X, self.theta)
        J = (-1 / m) * sum(self.y * np.log(h) + (1 - self.y) * np.log(1 - h))
        reg_J = J + self.Lambda / (2*m) * sum(self.theta ** 2)
        return reg_J

    def train(self):
        print('X shape:', self.X.shape)
        print('Theta Shape:', self.theta.shape)
        print('Y shape:', self.y.shape)

        for i in range(self.num_iter):
            m = self.y.shape[0]
            h = self.__sigmoid(self.X, self.theta)
            self.theta -= (self.learning_rate / m) * np.dot(self.X.T, (h - self.y))
            self.cost_history.append(self.__calculate_cost())
            if self.verbose:
                print("Batch: ", i + 1)
                print("Theta: ", self.theta)
                print("Cost: ", self.__calculate_cost())

    def plot_cost(self):
        plt.plot(self.cost_history[10:], '+')
        plt.xlabel('#Iteration')
        plt.ylabel('Cost')
        plt.show()

    def compare_to_Sklearn(self):
        model = LR()
        model.fit(self.X, self.y)
        print("My LR Theta:", self.theta)
        print("Sklearn LR Theta:", model.coef_)

    def predict(self, X):
        prediction = []
        for x in X:
            x = np.insert(x, 0, 1, axis=0)
            h = self.__sigmoid(x, self.theta)
            if h >= 0.5:
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction
