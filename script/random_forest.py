import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV as gscv
from sklearn import preprocessing

import toolkit


def get_train_data(file_train, scale=False, size=None):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    np.random.shuffle(data)

    x = data[:size,1:-1]
    if scale:
        x = preprocessing.scale(x)
    y = data[:size, -1]
    
    indices = toolkit.select_features(x, y, methods=('va', ))
    x = x[:, indices]
    return x, y, indices


def get_test_data(file_test, features, scale=False):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    
    x, ids = data[:,1:], data[:,0]
    if scale:
        x = preprocessing.scale(x)
    x = x[:, features]
    return x, ids 


def print_r(scale=True, size=3000):
    x, y, _features = get_train_data('../train3.dat', size=size*2)
    classifier = RandomForestClassifier(n_estimators=100,
                                        criterion='gini',
                                        n_jobs=-1,
                                        min_samples_split=10,
                                        max_features='sqrt',
                                        class_weight='balanced',)
    '''

    param_grid = {'min_samples_split': [2,5,10], 
            'class_weight':['balanced', 'balanced_subsample'], 
            'max_features': ['sqrt', 'log2', 0.9]}
    grid_s = gscv(classifier, 
                    param_grid=param_grid,
                    scoring='f1_weighted',
                    cv=10,
                    n_jobs=-1)
    grid_s.fit(x[:size],y[:size])
    print grid_s.best_estimator_
    print 'Best F1 score is: %s' % grid_s.best_score_

    best_c = grid_s.best_estimator_
    '''
    best_c = classifier
    cv_score = cross_val_score(best_c, 
                                x[size:], 
                                y[size:], 
                                cv=10, 
                                scoring='f1_micro',
                                n_jobs=-1,)
    best_c.fit(x[size:], y[size:])
    print cv_score
    print 'lb_score: %s' % toolkit.lb_scorer(best_c, x, y)


def train(x, y, depth=None):
    classifier = RandomForestClassifier(criterion='gini',
                                        n_estimators=100)

    param_grid = {'min_samples_split': [2,5,10], 
            'class_weight':['balanced', 'balanced_subsample'],
            'max_features':['sqrt', 0.9]} 
            
    grid_s = gscv(classifier, 
                    param_grid=param_grid,
                    scoring='f1_weighted',
                    cv=10,
                    n_jobs=-1)
    grid_s.fit(x, y)
    best_c = grid_s.best_estimator_

    print best_c
    print 'F1-Score of Best Validation Set is: %s' % grid_s.best_score_

    return best_c


def test(classifier, x):
    preds = classifier.predict(x)
    return preds


def main():
    x, y, features = get_train_data('../train3.dat')
    data, ids = get_test_data('../test3.dat', features)
    rfc = train(x, y)
    #evaluation(x,y,test_x,test_y)

    preds = test(rfc, data)
    probas = rfc.predict_proba(data)
    print 'Kaggle Score of all training set is: %s' % toolkit.lb_scorer(rfc, x, y)

    with open('../result/rf.v5.csv', 'w') as f:
        f.write('id,predict_0,predict_1,predict_2\n')
        for idx in xrange(probas.shape[0]):
            f.write('%s,%s\n' % (int(ids[idx]), ','.join([str(item) for item in probas[idx]])))

if __name__ == '__main__':
    #main()
    print_r()
    
    
