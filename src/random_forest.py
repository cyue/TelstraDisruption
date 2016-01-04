import sys
import numpy as np
import pandas as pd
import itertools
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn import cross_validation as cv
from sklearn.metrics import confusion_matrix


from StringIO import StringIO
import pydot

def samples(file_train, size=5000):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    x = data[:size,1:-1]
    y = data[:size, -1]
    return x, y


def train(x, y, depth=None):
    rfc = RandomForestClassifier(max_depth=depth)
    rfc.fit(x,y)
    
    scores = cv.cross_val_score(rfc, x, y, cv=6)
    return scores, rfc


def evaluation(x, y, test_x, test_y):
    ''' plot the result of evaluation concerning 
        1. spliting features 
        2. number of estimators
    '''
    max_feature_params = ['auto', 'log2', 1, 0.5, 0.99]
    confusion_matrixes = {}
    for max_feature in max_feature_params:
        rfc = RandomForestClassifier(max_features=max_feature)
        rfc.fit(x,y)

        pred_y = rfc.predict(test_x)
        confusion_matrixes[max_feature] = confusion_matrix(test_y, pred_y).ravel()

    confusion_df = pd.DataFrame(confusion_matrixes)

    f, ax = plt.subplots(figsize=(7, 5))
    confusion_df.plot(kind='bar', ax=ax)
    ax.legend(loc='best')
    ax.set_title("Guessed vs Correct (i, j) where i is the guess and j is the actual.")
    ax.grid()
    ax.set_xticklabels([str((i, j)) for i, j in list(itertools.product(range(3), range(3)))]);
    ax.set_xlabel("Guessed vs Correct")
    ax.set_ylabel("Correct")
    savefig('../img/random_forest_confusion_matrix.jpg')

    n_estimator_params = range(1,50)
    confusion_matrixes = {}
    for n_est in n_estimator_params:
        rfc = RandomForestClassifier(n_estimators=n_est)
        rfc.fit(x,y)
        pred_y = rfc.predict(test_x)
        confusion_matrixes[n_est] = confusion_matrix(test_y, pred_y)

        accuracy = lambda x: np.trace(x) / np.sum(x, dtype=float)
        confusion_matrixes[n_est] = accuracy(confusion_matrixes[n_est])
        accuracy_series = pd.Series(confusion_matrixes)
        f, ax = plt.subplots(figsize=(7, 5))
        accuracy_series.plot(kind='bar', ax=ax, color='k', alpha=.75)
        ax.grid()
        ax.set_title("Accuracy by Number of Estimators")
        ax.set_ylim(0, 1) # we want the full scope
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Number of Estimators")
    savefig('../img/rf_estimators.jpg')

def get_test_data(file_test):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)
    
    return data[:,1:]



def test(classifier, x):
    preds = classifier.predict(x)
    return x, preds


def main():
    x, y = samples('../train.csv', size=50000)
    test_x, test_y = samples('../train.csv', size=100)
    data = get_test_data('../test.csv')

    #evaluation(x,y,test_x,test_y)
    print train(x,y)[0]

if __name__ == '__main__':
    main()
    
    


