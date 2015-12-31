import sys
import numpy as np

from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import cross_validation as cv

def get_training_data(file_train):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    x = data[:,:-1]
    y = data[:,-1]
    return x, y

def get_test_data(file_test):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    return data

def train(x,y):
    classifier = dtc()
    scores = cv.cross_val_score(classifier, x, y, cv=10)
    
    classifier.fit(x, y)
    return classifier, scores

def test(classifier, x):
    preds = classifier.predict(x)
    return x, preds


def main():
    x, y = get_training_data('../train.csv')
    data = get_test_data('../test.csv')
    c, scores = train(x,y)
    x, preds = test(c,data)

    print scores

if __name__ == '__main__':
    main()
    

    
    


