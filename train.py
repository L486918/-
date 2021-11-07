import numpy as np
import pickle
import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == "__main__":
    x_in = open('dataset.pkl', 'rb')
    y_in = open('y.pkl', 'rb')
    x_set = pickle.load(x_in)
    y_set = pickle.load(y_in)
    x_in.close()
    y_in.close()
    x_set = np.array(x_set)
    y_set = np.array(y_set)
    train_X, valid_X, train_y, valid_y = train_test_split(x_set, y_set, test_size=0.33, random_state=5)
    print(train_y.shape[0], valid_y.shape[0])
    model = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=0), 5)
    model.fit(train_X, train_y)
    ver_pred = model.predict(valid_X).reshape((-1, 1))
    train_pred = model.predict(train_X).reshape((-1, 1))
    y_copy = np.copy(train_y)
    y_copy[train_pred == train_y] = 0
    y_copy[train_pred != train_y] = 1
    train_accuracy = 1 - sum(y_copy / y_copy.shape[0])
    y_copy = np.copy(valid_y)
    y_copy[ver_pred == valid_y] = 0
    y_copy[ver_pred != valid_y] = 1
    ver_accuracy = 1 - sum(y_copy / y_copy.shape[0])
    report = classification_report(valid_y, ver_pred, target_names=['non_face', 'face'])
    print(report)
    f = open('classifier_report.txt', 'w')
    f.write(report)
    f.close()
