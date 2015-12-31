import sys
import numpy as np

from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import cross_validation as cv

def get_training_data(file_train):
    data = np.genfromtxt(fname='../train.csv', delimiter=',', usecols=(1,2,3,4,5,6,7))
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    x = data[:,:-1]
    y = data[:,-1]
    return x, y

def get_test_data(file_test):
    data = np.genfromtxt(fname='../test.csv', delimiter=',', usecols=(1,2,3,4,5,6))
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    return data

def train(train_x, train_y, test_x
