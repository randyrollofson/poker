# Randy Rollofson
# CS545 Machine Learning
# Final Project
# Random Forest
# Poker Hand Analysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import random


# Suppresses sklearn warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn

spamdata = np.loadtxt("poker-hand-testing.data.txt", delimiter=",")
# spamdata = np.loadtxt("poker-hand-training-true.data.txt", delimiter=",")
# spamdata = np.loadtxt("spambase.data", delimiter=",")

X = np.delete(spamdata, 10, 1)
y = spamdata[:, 10]
accuracies1 = []
num_features_list = list(range(2, 11))
accuracies2 = []
full_list = list(range(10))

# Split data in to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Center training data
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_scaled = scaler.transform(X_train)

# Classify model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Take absolute value of weights
weight_list = clf.feature_importances_.copy()

# Sort indexes by highest weight value
idx = weight_list.argsort()
print("Least to most important indexes:", idx)

for i in range(2, 11):
    highest_idxs = np.delete(idx, np.s_[0:10 - i])
    X_trimmed = np.take(X, highest_idxs, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_trimmed, y, test_size=0.5)

    # scaler = preprocessing.StandardScaler().fit(X_train)
    # X_scaled = scaler.transform(X_train)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print("Accuracy with", i, "weights:", accuracy)
    accuracies1.append(accuracy)

print('\n')
for i in range(2, 11):
    random_idxs = random.sample(full_list, i)
    X_trimmed = np.take(X, random_idxs, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_trimmed, y, test_size=0.5)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print("Accuracy with", i, "weights:", accuracy)
    accuracies2.append(accuracy)

graph = plt.figure(1)
plt.axis([0, 10, 40, 70])
plt.title('Num Features vs Accuracy')
plt.xlabel('Number of Features Selected')
plt.ylabel('Accuracy (%)')
plt.plot(num_features_list, accuracies1)
plt.plot(num_features_list, accuracies2)
plt.legend(['Feature Selection', 'Random Selection'], loc='lower right')
plt.show()
