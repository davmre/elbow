import numpy as np
import tensorflow as tf
import bayesflow as bf


"""
Bayesian sparse matrix factorization with Gaussian mean-field posterior
"""

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

def mf_binary_sample_prior(nzr, nzc, prec, alpha, n, m, k):
    U = np.random.randn(n, k) * np.sqrt(alpha)
    V = np.random.randn(m, k) * np.sqrt(alpha)
    us = U[nzr, :]
    vs = V[nzc, :]
    rs = np.sum(us * vs, axis=1)
    probs, _ = bf.transforms.logit(rs * prec)
    nzz = np.random.rand(len(probs)) < probs
    return U, V, nzz

def construct_R(n, m, nzr, nzc, z):
    R = np.ones((n, m)) * np.nan
    nnz = len(nzr)
    for i in range(nnz):
        R[nzr[i], nzc[i]] = z[i]
    return R

################################################################

FIX_TRIANGLE, FIX_IDENTITY, FIX_NONE = np.arange(3)

def mf_binary_likelihood(U, V, nzr, nzc, nzz, noise_prec, alpha, n, m, k, fix_entries=FIX_TRIANGLE):
    
    with tf.name_scope("priors"):
        U_prior = tf.reduce_sum(bf.dists.gaussian_log_density(U, stddev=alpha), name="U_prior")
        V_prior = tf.reduce_sum(bf.dists.gaussian_log_density(V, stddev=alpha), name="V_prior")
    
    if fix_entries == FIX_IDENTITY:
        mask = np.float32(np.vstack((np.eye(k), np.ones((m-k, k)))))
        V = V * mask
    elif fix_entries == FIX_TRIANGLE:
        mask = np.float32(np.tril(np.ones((m, k))))
        V = V * mask
    else:
        pass
    
    with tf.name_scope("model"):
        Us = tf.gather(U, nzr, name="Us")
        #tf.histogram_summary("Us", Us)
        Vs = tf.gather(V, nzc, name="Vs")
        #tf.histogram_summary("Vs", Vs)
        Rs = tf.reduce_sum(tf.mul(Us, Vs), reduction_indices=1, name="Rs")
        #tf.histogram_summary("rs", Rs)

        probs, _ = bf.transforms.logit(Rs * noise_prec)
        
        #tf.histogram_summary("probs", probs)
        ll = tf.reduce_sum(bf.dists.bernoulli_log_density(nzz, probs), name="ll")
        joint_logprob = U_prior + V_prior + ll
        
    return joint_logprob


def run(n=100, m=100, k=5, alpha=1.0, prec=10.0, sparsity=0.3):

    # sample from the prior
    nzr, nzc = sample_sparsity(n,m, sparsity)
    U, V, z = mf_binary_sample_prior(nzr, nzc, prec, alpha, n, m, k)

    # define hparams for inference.
    # for right now, we'll do inference over the noise "precision" (really logit scaling)
    # but not the trait prior hyperparams
    alpha = np.ones((k,), dtype=np.float32)
    log_prec = tf.Variable(np.float32(0.0), name="log_prec")
    prec_var = tf.exp(log_prec)

    mf = MeanFieldInference(mf_binary_likelihood, 
                            nzr=nzr, nzc=nzc, nzz=np.float32(z), 
                            n=n, m=m, k=k, 
                            fix_entries=FIX_NONE,
                            alpha=alpha, noise_prec=prec_var)
    mf.add_latent("U", np.float32(np.random.randn(n, k)), np.float32(np.ones((n,k)) * 1e-3), None, shape=(n, k))
    mf.add_latent("V", np.float32(np.random.randn(m, k)), np.float32(np.ones((m,k)) * 1e-3), None, shape=(m, k))

    elbo = mf.build_stochastic_elbo(n_eps=1)

    d = {"density": mf.expected_density, "prec": prec_var}
    sess = tf.Session()
    mf.train(adam_rate=0.01, print_interval=10, display_dict=d, sess=sess)

if __name__ == "__main__":
    run()
