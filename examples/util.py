import numpy as np
import os
import urllib
import cPickle as pickle

def batch_generator(X, y, batch_size, max_steps=None):
    N = X.shape[0]
    i = 0
    while (max_steps is None) or i < max_steps:
        i += 1
        
        p = np.random.permutation(N)[:batch_size]
        xx = X[p]
        yy = y[p]
        yield i, xx, yy


def download(url):
    "download and return path to file"
    fname = os.path.basename(url)
    datadir = "downloads"
    datapath = os.path.join(datadir, fname)
    if not os.path.exists(datapath):
        print "downloading %s to %s"%(url, datapath)
        if not os.path.exists(datadir): os.makedirs(datadir)
        urllib.urlretrieve(url, datapath)
    return datapath

def fetch_dataset(url):
    datapath = download(url)
    fname = os.path.basename(url)
    extension =  os.path.splitext(fname)[-1]
    assert extension in [".npz", ".pkl"]
    if extension == ".npz":
        return np.load(datapath)
    elif extension == ".pkl":
        with open(datapath, 'rb') as fin:
            return pickle.load(fin)
    else:
        raise NotImplementedError

def get_mnist():
    # TODO: some of this code stolen from CGT
    mnist = fetch_dataset("http://rll.berkeley.edu/cgt-data/mnist.npz")
    Xdata = (mnist["X"]/255.).astype(np.float32)
    ydata = mnist["y"]
    return Xdata, ydata
