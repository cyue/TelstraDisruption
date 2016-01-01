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

    x = data[:size,:-1]
    y = data[:size, -1]
    return x, y


def train(x, y, depth=10):
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
        confusion_matrixes[max_feature] = confusion_matrix(test_y, pred_y).reshape(9)

        confusion_df = pd.DataFrame(confusion_matrixes)

        f, ax = plt.subplots(figsize=(7, 5))
        confusion_df.plot(kind='bar', ax=ax)
        ax.legend(loc='best')
        ax.set_title("Guessed vs Correct (i, j) where i is the guess and j is the actual.")
        ax.grid()
        ax.set_xticklabels([str((i, j)) for i, j in
                       list(itertools.product(range(2), range(2)))]);
        ax.set_xlabel("Guessed vs Correct")
        ax.set_ylabel("Correct")
        savefig('../img/random_forest_confusion_matrix.jpg')
        
    

def get_test_data(file_test):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    return data



def test(classifier, x):
    preds = classifier.predict(x)
    return x, preds


def main():
    x, y = samples('../train.csv', size=5000)
    test_x, test_y = samples('../train.csv', size=100)
    data = get_test_data('../test.csv')

    evaluation(x,y,test_x,test_y)

if __name__ == '__main__':
    main()
    
    


