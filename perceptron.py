import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True):
        """ Initialize class with chosen hyperparameters.
        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.initial_weights = None

    def fit(self, X, y, initial_weights=None, shuffle=False, deterministic=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        epoch_counter = 0

        bias = np.reshape([1] * X.shape[0], (X.shape[0], 1))
        X = np.concatenate((X, bias), axis=1)
        self.initial_weights = self.initialize_weights(X.shape[1]) if not initial_weights else initial_weights
        prev_weight = None

        # For each epoch
        while 1:

            # if shuffle, then shuffle the data
            if shuffle:
                X, y = self._shuffle_data(X, y)

            # For every array in training set
            for i in range(X.shape[0]):
                # do formula from class
                net = np.sum(np.multiply(X[i], self.initial_weights))
                output = 1 if net > 0 else 0

                # output does not match target
                if output != y[i]:
                    change_weight = self.lr * (y[i] - output) * X[i]
                    self.initial_weights = np.add(self.initial_weights, change_weight)

            epoch_counter += 1

            # If we want to do this deterministically
            if deterministic is not None:
                if epoch_counter >= deterministic:
                    break
            # If we want to do this until the change in weights is sufficiently small
            else:
                if prev_weight is None:
                    prev_weight = self.initial_weights
                else:
                    if np.sum(np.abs(np.subtract(np.abs(prev_weight), np.abs(self.initial_weights)))) < .5:
                        break
                    prev_weight = self.initial_weights

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        # init empty predictions
        predictions = np.zeros((X.shape[0], 1))

        # Do formula from class
        for i in range(X.shape[0]):
            net = np.sum(np.multiply(X[i], self.initial_weights))
            predictions[i] = 1 if net > 0 else 0

        return predictions

    def initialize_weights(self, count):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        return np.array([0.0] * count)

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        bias = np.reshape([1] * X.shape[0], (X.shape[0], 1))
        X = np.concatenate((X, bias), axis=1)

        predictions = self.predict(X)

        correct = 0.0
        for i in range(predictions.shape[0]):
            correct = correct + 1 if predictions[i] == y[i] else correct

        return correct / predictions.shape[0]

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        y = np.reshape(y, (X.shape[0], 1))
        tmp_array = np.concatenate((X, y), axis=1)
        np.random.shuffle(tmp_array)

        X = tmp_array[:, 0: tmp_array.shape[1] - 1]
        y = tmp_array[:, tmp_array.shape[1] - 1]

        return X, y

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.initial_weights
