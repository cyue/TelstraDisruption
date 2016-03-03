import sys
import numpy as np

from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

import toolkit


def get_train_data(file_train, scale=False, size=None):
    data = np.genfromtxt(fname=file_train, delimiter=',')
    np.random.shuffle(data)

    x = data[:size,1:-1]
    if scale:
        x = preprocessing.scale(x)
    y = data[:size, -1]
    indices = toolkit.select_features(x, y, methods=('variance', ))
    x = x[:,indices]
    return x, y, indices


def get_test_data(file_test, features,scale=False):
    data = np.genfromtxt(fname=file_test, delimiter=',') 
    
    x, ids = data[:,1:], data[:,0]
    if scale:
        x = preprocessing.scale(x)
    x = x[:, features]
    return x, ids 


def train(x,y):
    mlpc = MLPClassifier(batch_size=700,
                        learning_rate='invscaling',
                        shuffle=True,
                        max_iter=500,
                        activation='logistic',
                        alpha=0.01,
                        hidden_layer_sizes=(100,20),)
    '''
    param_grid = {'alpha': 10.0**-np.arange(1,4),
                    'hidden_layer_sizes': [(50,20),(100,10)]}
                    

    grid_s = GridSearchCV(mlpc, 
                        param_grid=param_grid,
                        cv=10,
                        scoring='f1_micro',
                        n_jobs=-1)

    grid_s.fit(x, y)
    best_c = grid_s.best_estimator_
    print 'F1 score of best is: %s' % grid_s.best_score_
    '''
    best_c = mlpc.fit(x,y)
    return best_c
 
def test(classifier, x):
    preds = classifier.predict(x)
    probas = classifier.predict_proba(x)
    return preds, probas


def eval(size=2000):
    x, y, _ = get_train_data('../train.dat', size=2*size)
    mlpc = MLPClassifier(batch_size=700,
                        learning_rate='invscaling',
                        shuffle=True,
                        max_iter=500,
                        activation='logistic',
                        hidden_layer_sizes=(100,10),)

    param_grid = {'alpha': 10.0**-np.arange(1,4),}
                    

    grid_s = GridSearchCV(mlpc, 
                        param_grid=param_grid,
                        cv=6,
                        scoring='f1_micro',
                        n_jobs=-1)
    
    grid_s.fit(x[:size],y[:size])
    print grid_s.best_estimator_
    print 'Best F1 score is: %s' % grid_s.best_score_

    best_c = grid_s.best_estimator_
    cv_score = cross_val_score(best_c, 
                                x[size:], 
                                y[size:], 
                                cv=6, 
                                scoring='f1_micro',
                                n_jobs=-1,)

    print 'F1 score of the rest is: %s' % cv_score
    print 'Kaggle score of all is: %s' % toolkit.lb_scorer(best_c, x, y)

def main():
    x, y, features = get_train_data('../train_discard_res_severity.dat')
    data, ids = get_test_data('../test_discard_res_severity.dat', features)

    mlpc = train(x, y)
    preds, probas = test(mlpc, data)
    print 'Kaggle score of all training set: %s' % toolkit.lb_scorer(mlpc, x, y)

    with open('../result/mlp.v5.csv', 'w') as f:
        f.write('id,predict_0,predict_1,predict_2\n')
        for idx in xrange(preds.shape[0]):
            f.write('%s,%s\n' % (int(ids[idx]), ','.join([np.str(item) for item in probas[idx]])))
    


    print 'Kaggle score of all is: %s' % toolkit.lb_scorer(mlpc, x, y)


if __name__ == '__main__':
    #eval()
    #print 'Main Function begin'
    main()
    
