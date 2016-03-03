import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV as gscv

import toolkit


def get_train_data(file_train, scale=False, size=None):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    np.random.shuffle(data)

    x = data[:size,1:-1]
    if scale:
        x = preprocessing.scale(x)
    y = data[:size, -1]
    indices = toolkit.select_features(x, y, methods=('variance', ))
    x = x[:, indices]
    return x, y, indices


def get_test_data(file_test, features, scale=False):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    
    x, ids = data[:,1:], data[:,0]
    if scale:
        x = preprocessing.scale(x)
    x = x[:, features]
    return x, ids 


def train(x,y):
    gbtc = GradientBoostingClassifier(max_features=0.9, 
                                        max_depth=3,
                                        min_samples_split=2,
                                        n_estimators=300,
                                        learning_rate=0.05,)
    '''
    param_grid = {'max_features': ['sqrt', 'log2', 0.9],
                    'min_samples_split': [2,5,10]}
    
    grid_s = gscv(gbtc, 
                param_grid=param_grid, 
                cv=10, 
                scoring='f1_micro',
                n_jobs=-1)
    grid_s.fit(x,y)
 
    best_c = grid_s.best_estimator_
    print best_c
    print 'Best F1-score: %s' % grid_s.best_score_
    '''
    best_c = gbtc
    best_c.fit(x,y)
    return best_c

def test(classifier, x):
    preds = classifier.predict(x)
    probas = classifier.predict_proba(x)
    return preds, probas


def eval(size=3000):
    x, y, features = get_train_data('../train_discard_res_severity.dat', size=2*size)
    gbtc = GradientBoostingClassifier(max_features='auto', 
                                        max_depth=3,
                                        n_estimators=300,
                                        learning_rate=0.05)

    param_grid = {'min_samples_split': [2,5,10],
                    'max_features': ['sqrt', 'log2', 0.9],}
    
    grid_s = gscv(gbtc, 
                param_grid=param_grid, 
                cv=6, 
                scoring='f1_micro',
                n_jobs=-1)
    grid_s.fit(x[:size], y[:size])
    
    best_c = grid_s.best_estimator_
    print best_c
    print 'Best F1-score: %s' % grid_s.best_score_

    
    cv_f1_score = cross_val_score(best_c,
                                x[size:], 
                                y[size:],
                                cv=6,
                                scoring='f1_micro',
                                n_jobs=-1)
    print 'CV F1-score of the rest: %s' % cv_f1_score 

    print 'Kaggle score of all: %s' % toolkit.lb_scorer(best_c, x, y)


def main():
    x, y, features = get_train_data('../train_discard_res_severity.dat')
    data, ids = get_test_data('../test_discard_res_severity.dat',features)

    gbtc = train(x, y)
    preds, probas = test(gbtc, data)
    print 'Kaggle score of all training set: %s' % toolkit.lb_scorer(gbtc, x, y)

    with open('../result/gbtc.v5.csv', 'w') as f:
        f.write('id,predict_0,predict_1,predict_2\n')
        for idx in xrange(preds.shape[0]):
            f.write('%s,%s\n' % (int(ids[idx]), ','.join([np.str(item) for item in probas[idx]])))
    

if __name__ == '__main__':
    eval()
    #main()
