import sys
import numpy as np


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import preprocessing 
from sklearn import cross_validation as CV

def train(x, y, scale=True):
    if scale:
        x = preprocessing.scale(x)
    classifier = KNN(n_neighbors = 50, algorithm = 'kd_tree', 
                        weights='distance', n_jobs=-1)
    scores = CV.cross_val_score(classifier, x, y, cv=10)

    return scores, classifier

def test(x, y, scale=True):
    pass


def get_train_data(file_train):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    # split log volumn to group samples based on frequency
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    # remove id column
    x = data[:,1:-1]
    y = data[:,-1]

    return x, y


def main():
    x,y = get_train_data('../train.csv')
    scores, clc = train(x,y)

    print scores


if __name__ == '__main__':
    main()
