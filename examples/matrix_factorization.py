import numpy as np
import tensorflow as tf

from elbow import Gaussian, Model

from elbow.models.factorizations import NoisyGaussianMatrixProduct, NoisySparseGaussianMatrixProduct

####################################################################
# methods to sample from a Gaussian matrix prior with uniform random
# sparsity pattern.

def sample_sparsity(n, m, p=0.1):
    # sample a sparsity mask for an nxm matrix
    if p == 1:
        unique_pairs = [(i,j) for i in range(n) for j in range(m)]
    else:
        nnz = int(n*m*p)
        rows = np.random.randint(0,n,nnz)
        cols = np.random.randint(0,m,nnz)
        unique_pairs = set(zip(rows, cols))
    nzr, nzc = zip(*unique_pairs)
    return np.asarray(nzr), np.asarray(nzc)

def construct_R(n, m, nzr, nzc, z):
    R = np.ones((n, m)) * np.nan
    nnz = len(nzr)
    for i in range(nnz):
        R[nzr[i], nzc[i]] = z[i]
    return R

################################################################

def sparse_model(row_idxs, col_idxs, n=10, m=9, prior_std=1.0, noise_std=0.1):

    A = Gaussian(mean=0.0, std=prior_std, shape=(n, 3), name="A")
    B = Gaussian(mean=0.0, std=prior_std, shape=(m, 3), name="B")
    C = NoisySparseGaussianMatrixProduct(A=A, B=B,
                                         std=noise_std,
                                         row_idxs=row_idxs,
                                         col_idxs=col_idxs,
                                         name="C")

    jm = Model(C)

    return jm

def main():

    """
    Sample data from the prior
    """
    n = 100
    m = 50
    k = 3
    sparsity = 0.2
    nzr, nzc = sample_sparsity(n,m, sparsity)
    print "sampled sparsity pattern has %d of %d entries nonzero" % (len(nzr), n*m)
    
    jm = sparse_model(nzr, nzc, n=n, m=m)
    sampled = jm.sample()
    jm["C"].observe(sampled["C"])

    """
    Reconstruction err for true latent traits.
    """
    sA = sampled["A"]
    sB = sampled["B"]
    sC = np.dot(sA, sB.T)
    mean_abs_err = np.mean(np.abs(sC[nzr,nzc] - sampled["C"]))
    print "true latent values reconstruct observations with mean deviation %.3f" % (mean_abs_err)

    """
    Run inference and compute reconstruction err on the inferred
    traits. Note we consider reconstruction error on C, rather than
    direct recovery of the latents A and B, because due to model
    symmetries the latter will only be recovered up to a linear
    transformation.
    """

    jm.train(avg_decay=0.995)
    posterior = jm.posterior()
    
    qA = posterior["q_A"]["mean"]
    qB = posterior["q_B"]["mean"]
    qC = np.dot(qA, qB.T)
    mean_abs_err_inferred = np.mean(np.abs(qC[nzr,nzc] - sampled["C"]))
    print "inferred latent values reconstruct observations with mean deviation %.3f" % (mean_abs_err_inferred)


if __name__ == "__main__":
    main()

