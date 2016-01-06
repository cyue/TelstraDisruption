import sys
import numpy as np

from sklearn.linear_model import LogisticRegression 
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation as CV
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


def train(x, y):
    ''' according to fault distribution, '0' counts 60%, '1' counts 28%, '2' counts 12%
    '''
    mlc = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)
    scores = CV.cross_val_score(mlc, x, y, cv=10)
    return scores


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
    
    
def get_training_data(file_train, with_scale=True):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    # split log volumn to group samples based on frequency
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    # remove id column
    x = data[:,1:-1]
    y = data[:,-1]

    if with_scale:
       x = preprocessing.normalize(x) 
    return x, y

def main():
    x, y = get_training_data('../train.csv')
    scores = train(x,y)
    print '10-folds cross validation scores without considering label distribution: \n%s' % scores
    
    classifiers = get_classifiers(x,y)
    scores = cross_val_score(classifiers, x, y)
    
    print '10-folds cross validation scores with label distribution: \n%s' % scores

if __name__ == '__main__':
    main()

