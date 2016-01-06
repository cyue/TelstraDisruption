import sys
import numpy as np
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig

from sklearn.decomposition import PCA
from sklearn import preprocessing


def get_data(file_data):
    data = np.genfromtxt(fname=file_data, delimiter=',')
    # split log volumn to group samples based on frequency
    data[:,6] = np.floor(data[:,6]/10)
    np.random.shuffle(data)

    # remove id column
    x = data[:,1:-1]
    y = data[:,-1]
    return x,y


def plot(x,y):
    pca = PCA(n_components=1)
    x_prime = pca.fit_transform(x)
    
    print pca.explained_variance_ratio_

    x_axis = np.arange(x_prime.shape[0])
    y_axis = x_prime.ravel()
    plt.figure()
    scatter = plt.scatter(x_axis, y_axis, marker='*', linewidth=0.1)
    savefig('../img/1-d_distribution.jpg')
    plt.close('all')

    x_comb = np.insert(x, 0, y, axis=1)
    for label in np.unique(y):
        x_prime = pca.fit_transform(x_comb[x_comb[:,0]==label, 1:])
        print label, pca.explained_variance_ratio_
        x_axis = np.arange(x_prime.shape[0])
        y_axis = x_prime.ravel()
        plt.figure()
        scatter = plt.scatter(x_axis, y_axis, marker='o', c='r', linewidth=0.1)
        savefig('../img/1-d class_%s.jpg' % label)
        plt.close('all')


def plot_pdf(x,y):
    pca = PCA(n_components=1)
    x_comb = np.insert(x, 0, y, axis=1)
    for label in np.unique(y):
        # iterate and select data with exact label
        data_prime = pca.fit_transform(x_comb[x_comb[:,0]==label, 1:])
        # sort then fit
        data_prime_sort = np.sort(data_prime.ravel())
        N = len(data_prime_sort)

        x_axises = np.arange(data_prime_sort[0], data_prime_sort[-1], 10)
        y_axises = np.arange(np.float(len(x_axises)))
        for idx in xrange(len(x_axises)):
            axis = x_axises[idx]
            if idx < len(x_axises)-1:
                next_axis = x_axises[idx+1]
            else:
                next_axis = data_prime_sort[-1]
            # y_axis is probability at x_axis
            y_axises[idx] = len(np.intersect1d(data_prime_sort[data_prime_sort > axis],
                                data_prime_sort[data_prime_sort <= next_axis])) / np.float(N)
            
        plt.figure()
        scatter = plt.scatter(x_axises, y_axises, marker='o', c='r', linewidth=1.0)
        savefig('../img/1-d class_%s.jpg' % np.int(label))
        plt.close('all')
        
        
def main():
    x, y = get_data('../train.csv')
    plot_pdf(x,y)


if __name__ == '__main__':
    main()
    
