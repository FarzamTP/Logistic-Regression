import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

X = []
labels = []

with open('normalized_data.txt', 'r') as f:
    for line in f.readlines():
        x, y, label = (line.split(','))
        X.append([float(x), float(y)])
        labels.append(label)
        print(line)

X = np.asarray(X)

new_X = preprocessing.normalize(X)

# plt.hist(X, density=True)
# plt.ylabel('Probability')
# plt.show()
#
# plt.hist(new_X, density=True)
# plt.ylabel('Probability')
# plt.show()

with open('normalized_data.txt', 'w+') as f:
    for index, line in enumerate(new_X):
        f.write(str(line[0]) + ',' + str(line[1]) + ',' + labels[index])
