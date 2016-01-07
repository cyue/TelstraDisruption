import sys
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import cross_validation as cv
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score

from StringIO import StringIO
import pydot

import toolkit

def get_training_data(file_train):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    # split log volumn to group samples based on frequency
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    # remove id column
    x = data[:,1:-1]
    y = data[:,-1]
    return x, y


def get_test_data(file_test):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    return data[:,1:], data[:,0]


def train(x,y,depth=None, scale=True):
    if scale:
        x = preprocessing.scale(x)
    
    classifier = dtc(max_depth=depth, class_weight='balanced')
    
    classifier.fit(x, y)
    ne0 = classifier.feature_importances_ != 0
    y_imp = classifier.feature_importances_[ne0]
    x_imp = np.arange(len(classifier.feature_importances_))[ne0]
    return classifier, x_imp, y_imp


def test(classifier, x):
    preds = classifier.predict(x)
    return preds


def decision_graph(classifier):
    str_buffer = StringIO()
    tree.export_graphviz(classifier, out_file=str_buffer)
    graph = pydot.graph_from_dot_data(str_buffer.getvalue())
    graph.write("../img/decision_graph.jpg")


def print_r(scale=True, size=10000):
    x, y = get_training_data('../train.csv')
    if scale:
        x = preprocessing.scale(x)

    classifier = dtc(class_weight='balanced')
    score = cv.cross_val_score(classifier, x, y, cv=10, scoring='roc_auc')
    
    #classifier = dtc(class_weight='balanced')
    #classifier.fit(x[:size],y[:size])
    #preds = classifier.predict(x[size:])
    
    #f1 = f1_score(y[size:], preds, labels=[0.0,1.0,2.0], average='weighted')
    

    print score


def main():
    x, y = get_training_data('../train.csv')
    data, ids = get_test_data('../test.csv')
    c, x_imp, y_imp = train(x,y)
    preds = test(c, data)
    #decision_graph(c)

    print 'id,predict_0,predict_1,predict_2'
    for tid, label in toolkit.vote(ids, preds):
        print '%s,%s' % (tid, ','.join([np.str(item) for item in label]))

if __name__ == '__main__':
    #main()
    print_r()
    

