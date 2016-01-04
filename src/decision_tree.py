import sys
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import cross_validation as cv

from StringIO import StringIO
import pydot

def get_training_data(file_train):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    x = data[:,1:-1]
    y = data[:,-1]
    return x, y


def get_test_data(file_test):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    return data[:,1:]


def train(x,y,depth=10):
    classifier = dtc(max_depth=depth, presort=True)#criterion='entropy')
    scores = cv.cross_val_score(classifier, x, y, cv=10)
    
    classifier.fit(x, y)
    ne0 = classifier.feature_importances_ != 0
    y_imp = classifier.feature_importances_[ne0]
    x_imp = np.arange(len(classifier.feature_importances_))[ne0]
    return classifier, scores, x_imp, y_imp


def test(classifier, x):
    preds = classifier.predict(x)
    return x, preds


def decision_graph(classifier):
    str_buffer = StringIO()
    tree.export_graphviz(classifier, out_file=str_buffer)
    graph = pydot.graph_from_dot_data(str_buffer.getvalue())
    graph.write("../img/decision_graph.jpg")


def main():
    x, y = get_training_data('../train.csv')
    data = get_test_data('../test.csv')
    c, scores, x_imp, y_imp = train(x,y)
    x, preds = test(c,data)
    decision_graph(c)

    print scores
    print x_imp
    print y_imp

if __name__ == '__main__':
    main()
    

    
    


