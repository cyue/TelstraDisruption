import sys
import numpy as np
import toolkit

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV 
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

def get_classifiers(x, y):
    # classifier for 0
    lgr0 = LogisticRegression(class_weight={0:.4, 1:.6})

    # classifier for 1
    lgr1 = LogisticRegression(class_weight={0:.72, 1:.28})

    # classifier for 2
    lgr2 = LogisticRegression(class_weight={0:.88, 1:.12})

    return lgr0, lgr1, lgr2


def cross_val_score(estimators, x, y, cv=10):
    ''' label 0, 1, 2 
    '''
    kf = StratifiedKFold(y, n_folds=cv)
    scores = []
    for train_idx, test_idx in kf:
        x_train, x_test= x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # initialize final predictions
        max_prob, class_label = np.zeros(len(x_test)), np.zeros(len(x_test))
        for label, estimator in enumerate(estimators):
            _y = (y_train == label).astype(int)
            estimator.fit(x_train, _y)
            probas = estimator.predict_proba(x_test)[:,1]
            for idx, proba in enumerate(probas):
                if proba > max_prob[idx]:
                    max_prob[idx] = proba
                    class_label[idx] = label
        # validate the result to scores
        score = (y_test == class_label).mean()
        scores.append(score)

    return scores
    
def selfmain():
    x, y = get_training_data('../train.csv')
    scores = train(x,y)
    print '10-folds cross validation scores without considering label distribution: \n%s' % scores
    
    classifiers = get_classifiers(x,y)
    scores = cross_val_score(classifiers, x, y)
    
    print '10-folds cross validation scores with label distribution: \n%s' % scores



def get_training_data(file_train, scale=True, size=None):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    # split log volumn to group samples based on frequency
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    # remove id column
    x = data[:size,1:-1]
    if scale:
        x = preprocessing.scale(x)
    y = data[:size,-1]
    return x, y


def get_test_data(file_test, scale=True):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    data[:,6] = np.floor(data[:,6]/10)

    x, ids = data[:,1:], data[:,0]
    if scale:
        x = preprocessing.scale(x)

    return x, ids


def print_r(scale=True, size=10000):
    x, y = get_training_data('../train.csv')
    test, ids = get_test_data('../test.csv')
    
    lgr = LogisticRegressionCV(cv=10, class_weight='balanced',
            solver='lbfgs', multi_class='ovr', scoring='f1_weighted')
    mlc = OneVsRestClassifier(lgr, n_jobs=-1)
    mlc.fit(x[:size], y[:size])

    preds = mlc.predict(x[size:2*size])
    cm = confusion_matrix(y[size:2*size], preds)

    print mlc.score(x[size:2*size], y[size:2*size])
    print cm


def train(x, y):
    lgr = LogisticRegressionCV(cv=10, class_weight='balanced',
            solver='lbfgs', multi_class='ovr', scoring='f1_weighted')
    mlc = OneVsRestClassifier(lgr, n_jobs=-1)
    mlc.fit(x,y)
    return mlc


def test(classifier, x):
    preds = classifier.predict(x)
    return preds
    

def main():
    x, y = get_training_data('../train.csv')
    data, ids = get_test_data('../test.csv')
    c = train(x,y)
    preds = test(c, data)

    print 'id,predict_0,predict_1,predict_2'
    for tid, label in toolkit.vote(ids, preds):
        print '%s,%s' % (tid, ','.join([np.str(item) for item in label]))
if __name__ == '__main__':
    main()
    #print_r()

