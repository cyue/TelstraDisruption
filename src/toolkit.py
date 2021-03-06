import numpy as np
import sys


def kaggle_scorer(y_truth, y_preds, proba):
    n = len(y_truth)
    m = np.unique(y_truth)
    y_truth_matrix = np.zeros(m*n).reshape(n,m)
    for idx, y in enumerate(y_truth):
        y_truth_matrix[idx,y] = 1
        
    loss = -1./n * np.sum(y_truth_matrix * np.log(proba))
    return loss



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
        
