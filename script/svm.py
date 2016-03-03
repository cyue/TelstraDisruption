import sys
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn import cross_validation as CV
from sklearn import preprocessing

import toolkit


def get_train_data(file_train, scale=False, size=None):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    np.random.shuffle(data)

    x = data[:size,1:-1]
    if scale:
        x = preprocessing.scale(x)
    y = data[:size, -1]
    return x, y


def get_test_data(file_test, scale=False):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    
    x, ids = data[:,1:], data[:,0]
    if scale:
        x = preprocessing.scale(x)
    return x, ids 


def train(x,y, classifier):
    classifier.fit(x,y)
    return classifier

def test(classifier, x):
    preds = classifier.predict(x)
    probas = classifier.predict_proba(x)
    return preds, probas


def eval(size=3000):
    x, y = get_train_data('../train.dat')
    svc = SVC(C=0.9, cache_size=300, class_weight='balanced',
                decision_function_shape='ovr') 
    bc = BaggingClassifier(svc, n_estimators=20, max_samples=0.5, max_features=0.95,
            n_jobs=-1)
    score = CV.cross_val_score(bc, x[:size], y[:size], cv=6, scoring='accuracy', n_jobs=-1)
    
    bc.fit(x[:size], y[:size])
    preds = bc.predict(x[size:2*size])
    probas = bc.predict_proba(x[size:2*size])
    lb_score = toolkit.kaggle_scorer(y[size:2*size], preds, probas)
    
    print score
    print lb_score

def main():
    x, y = get_train_data('../train.dat')
    data, ids = get_test_data('../test.dat')

    svc = SVC(C=0.9, cache_size=300, class_weight='balanced',
                decision_function_shape='ovr', probability=True)
    bc = BaggingClassifier(svc, n_estimators=50, max_samples=0.2, max_features=0.95,
            n_jobs=-1, warm_start=True)
    bc = train(x,y,bc)
    preds, probas = test(bc, data)

    print 'id,predict_0,predict_1,predict_2'
    for idx in xrange(preds.shape[0]):
        print '%s,%s' % (int(ids[idx]), ','.join([np.str(item) for item in probas[idx]]))

if __name__ == '__main__':
    #eval()
    main()
    
