import pickle
import numpy as np


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_wearkers_limit = n_weakers_limit
        self.best_model = []

    def is_good_enough(self, error_rate):
        '''Optional'''

    def error_rate(self, predict, y, weight):
        y_copy = np.copy(y)
        y_copy[predict == y] = 0
        y_copy[predict != y] = 1
        return np.multiply(y_copy, weight.T).sum()

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_sample = X.shape[0]
        weight = np.ones((n_sample, 1)) / n_sample  ## 初始权重一样
        for i in range(self.n_wearkers_limit):
            weaker_classifier = self.weak_classifier
            weaker_classifier.fit(X, y, sample_weight=[i for item in weight for i in item])
            predict = weaker_classifier.predict(X).reshape(-1, 1)  ## 转为一列
            error_rate = self.error_rate(predict, y, weight.T)
            if error_rate == 0:
                break
            alpha = np.log((1 - error_rate) / error_rate) / 2
            Z = np.multiply(weight, np.exp(-alpha * np.multiply(predict, y))).sum()
            weight = np.multiply(weight, np.exp(-alpha * np.multiply(predict, y))) / Z
            self.best_model.append((alpha, weaker_classifier))

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        sample_num = X.shape[0]
        predict_scores = []
        score = 0
        for i in range(len(self.best_model)):
            predict_scores = predict_scores.append(self.best_model[i][0] * self.best_model[i][1].predict(X))
        score = predict_scores.sum(axis=0)
        return score

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        pred = np.zeros((X.shape[0], 1))
        for item in self.best_model:
            pred += item[0] * ((item[1].predict(X)).reshape(-1, 1))
        pred[pred > threshold] = 1
        pred[pred <= threshold] = -1

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
