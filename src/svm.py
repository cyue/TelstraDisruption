import sys
import numpy as np
import toolkit

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation as CV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

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


def train(x, y):
    svc = SVC(C=0.1, kernel='rbf', class_weight='balanced', 
            cache_size = 1000, decision_function_shape='ovr')
    svc = OneVsRestClassifier(svc, n_jobs=-1)
    svc.fit(x, y)

    return svc


def print_r(scale=True, size=10000):
    x, y = get_training_data('../train.csv')
    test, ids = get_test_data('../test.csv')

    svc = SVC(C=0.1, kernel='rbf',
                cache_size = 1000, decision_function_shape='ovr')
    svc = OneVsRestClassifier(svc)
    scores = CV.cross_val_score(svc, x[:size], y[:size], cv=5, scoring='f1_weighted', n_jobs=-1)
    svc.fit(x[:size],y[:size])

    preds = svc.predict(x[size:2*size])
    cm = confusion_matrix(y[size:2*size], preds)

    print scores
    print cm
    

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


