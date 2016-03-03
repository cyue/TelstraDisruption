import numpy as np
import sys

from sklearn.feature_selection import SelectPercentile as SP
from sklearn.feature_selection import VarianceThreshold as VT
from sklearn.feature_selection import SelectFromModel as SFM
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression as LGR


def select_features(x, y, methods=('variance', 'correlation', 'l1', 'forest')):
    ''' methods = ('variance', 'correlation', 'l1', 'forest')
        - variance: use variance threshold to discard features that are mostly 0 or 1
        - correlation: use chi2 test to remove most very correlated features
        - l1: use l1 penalty to remove features that make solution sparse
        - forest: use ExtraTreesClassifier to point out importance of features
                    select important ones
    '''
    idx_list = []
    x_indices = set(np.arange(x.shape[1]))

    if 'variance' in methods:
        vt = VT(threshold=(0.99*(1-0.99)))
        vt.fit(x)
        variance_indices = set(vt.get_support(indices=True))
        idx_list.append(variance_indices)
        print 'variance: %s' % len(variance_indices)

    if 'correlation' in methods:
        cr = SP(chi2, percentile=80)
        cr.fit(x,y)
        correlation_indices = set(cr.get_support(indices=True))
        idx_list.append(correlation_indices)
        print 'corrlation: %s' % len(correlation_indices)

    if 'l1' in methods:
        lgr = LGR(C=0.9, penalty='l1').fit(x,y)
        model = SFM(lgr, prefit=True)
        l1_indices = set(model.get_support(indices=True))
        idx_list.append(l1_indices)
        print 'l1: %s' % len(l1_indices)

    if 'forest' in methods:
        clf = ExtraTreesClassifier(n_estimators=100, max_features=0.8,n_jobs=-1).fit(x,y)
        forest = SFM(clf, prefit=True)
        forest_indices = set(forest.get_support(indices=True))
        idx_list.append(forest_indices)
        print 'forest: %s' % len(forest_indices)

    for indices in idx_list:
        x_indices = x_indices & indices
    print 'All: %s' % len(x_indices)

    return list(x_indices)
        

def kaggle_scorer(y_truth, y_preds, probas):
    n = len(y_truth)
    m = len(np.unique(y_truth))
    y_truth_matrix = np.zeros(m*n).reshape(n,m)
    
    for idx, y in enumerate(y_truth):
        y_truth_matrix[idx,y] = 1

    for rowidx, proba in enumerate(probas):
        for colidx, i in enumerate(proba):
            if i == 0.:
                probas[rowidx][colidx] = 0.1
        
    loss = -1./n * np.sum(y_truth_matrix * np.log(proba))
    return loss


def lb_scorer(estr, x, y_truth):
    n = len(y_truth)
    m = len(np.unique(y_truth))
    y_truth_matrix = np.zeros(m*n).reshape(n,m)

    probas = estr.predict_proba(x)
    
    for idx, y in enumerate(y_truth):
        y_truth_matrix[idx,y] = 1

    for rowidx, proba in enumerate(probas):
        for colidx, i in enumerate(proba):
            if i == 0.:
                probas[rowidx][colidx] = 0.1
        
    loss = -1./n * np.sum(y_truth_matrix * np.log(proba))
    optimal = -1 * loss
    return optimal



def transform(ids, preds):
    pred_matrix = np.zeros(3*len(preds)).reshape(len(preds),3)
    for row_idx in xrange(preds.shape[0]):
        pred_matrix[row_idx, preds[row_idx]] = 1
        yield int(ids[row_idx]), pred_matrix[row_idx].astype(int)



def vote(ids, predicts):
    ''' vote the majority of predicts in one id 
    '''
    ids = [np.int(id) for id in ids]
    predicts = [np.int(p) for p in predicts]
    id_dict = {}
    for idx, id in enumerate(ids):
        if id not in id_dict:
            preds = [0,0,0]
            preds[predicts[idx]] = 1
            id_dict.setdefault(id, preds)
        else:
            id_dict[id][predicts[idx]] = id_dict[id][predicts[idx]] + 1
    
    for id in id_dict:
        # return id, the index(label) which is the majority
        label_list = [item / np.max(id_dict[id]) for item in id_dict[id]]
        if np.sum(label_list) > 1:
            # randomly set one to 0
            label_list[label_list.index(1)] = 0
            
        yield id, label_list
        
