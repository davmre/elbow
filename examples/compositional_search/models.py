import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import FlatDistribution
from bayesflow.models.elementary import GaussianMatrix, BernoulliMatrix, GammaMatrix, BetaMatrix, DirichletMatrix
from bayesflow.models.q_distributions import DeltaQDistribution, GaussianQDistribution, BernoulliQDistribution
from bayesflow.models.matrix_decompositions import *
from bayesflow.models.transforms import PointwiseTransformedMatrix, PointwiseTransformedQDistribution, Transpose


from grammar import list_structures

def build_column_stds(shape, settings, name):
    N, K = shape
    
    alpha = settings.gaussian_colprec_alpha
    beta = settings.gaussian_colprec_beta

    if settings.gaussian_ard:
        prec_dim = K
    else:
        prec_dim = 1
        
    prec = GammaMatrix(alpha=alpha, beta=beta, output_shape=(prec_dim,),
                            dtype=np.float32, name="%s_precs" % name)
    std = PointwiseTransformedMatrix(prec, bf.transforms.reciprocal_sqrt,
                                     name="%s_std" % name)

    q1 = GaussianQDistribution(shape=(prec_dim,))
    qprecs = PointwiseTransformedQDistribution(q1, bf.transforms.exp)
    prec.attach_q(qprecs)

    return std
    
def build_gaussian(shape, settings, name):
    N, K = shape

    col_stds = build_column_stds(shape, settings, name)
    
    G = GaussianMatrix(mean=0.0, std=col_stds, name=name, output_shape=shape)
    
    return G

def build_bernoulli(shape, settings, name):
    N, K = shape
    a, b = settings.beta_prior_params
    
    pi = BetaMatrix(alpha=a, beta=b, output_shape=(K,), name="%s_pi" % name)
    q1 = GaussianQDistribution(shape=(K,))
    qpi = PointwiseTransformedQDistribution(q1, bf.transforms.logit)
    pi.attach_q(qpi)

    B = BernoulliMatrix(p=pi, output_shape=(N, K), name=name)
    q_B = BernoulliQDistribution(shape=(N, K))
    B.attach_q(q_B)

    return B

def build_transpose(A, name):
    return Transpose(A, name=name)

def build_noise_std(settings, name):
    if settings.constant_noise_std is not None:
        noisestd = settings.constant_noise_std
    else:
        alpha = settings.noise_prec_alpha
        beta = settings.noise_prec_beta
        noiseprec = GammaMatrix(alpha=alpha, beta=beta, output_shape=(1,),
                                dtype=np.float32, name="%s_noise_precs" % name)
        noisestd = PointwiseTransformedMatrix(noiseprec, bf.transforms.reciprocal_sqrt,
                                              name="%s_noise_std" % name)
        q1 = GaussianQDistribution(shape=(1,))
        qprecs = PointwiseTransformedQDistribution(q1, bf.transforms.exp)
        noiseprec.attach_q(qprecs)
        
    return noisestd

def build_lowrank(G1, G2, settings, name):
    noise_std = build_noise_std(settings, name)
    G1.attach_gaussian_q()
    G2.attach_gaussian_q()
    return NoisyGaussianMatrixProduct(G1, G2, noise_std, name=name)

def build_features(B, G, settings, name):
    noise_std = build_noise_std(settings, name)

    # assume B already has a Q attached, since it can't exist in isolation
    G.attach_gaussian_q()
    return NoisyLatentFeatures(B, G, noise_std, name=name)

def build_chain(G, settings, name):
    noise_std = build_noise_std(settings, name)
    G.attach_gaussian_q()
    return NoisyCumulativeSum(G, noise_std, name=name)

def build_sparsity(G, settings, name):
    expG = PointwiseTransformedMatrix(G1, bf.transforms.exp, name="%s_exp" % name)
    
    stds = build_column_stds(G.output_shape, settings, name)
    return MultiplicativeGaussianNoise(expG1, stds, name=name)

def build_clustering(centers, shape, settings, name):
    N, D = shape
    K, D2 = centers.output_shape
    assert(D==D2)
    
    weights = DirichletMatrix(alpha=settings.dirichlet_alpha,
                              output_shape=(K,),
                              name="%s_weights" % name)

    q1 = GaussianQDistribution(shape=(K,))
    qweights = PointwiseTransformedQDistribution(q1, bf.transforms.simplex)
    weights.attach_q(qweights)

    q_centers = centers.attach_gaussian_q()
    
    noise_std = build_noise_std(settings, name)
    return GMMClustering(weights=weights, centers=centers,
                         std=noise_std, output_shape=shape, name=name)

def build_model(structure, shape, settings, tree_path="g"):

    N, D = shape
    K = settings.max_rank
    
    if isinstance(structure, str):
        if structure == "g":
            return build_gaussian(shape, settings, name="g_%s" % tree_path)
        elif structure == 'b':
            return build_bernoulli(shape, settings, name="b_%s" % tree_path)
        else:
            raise Exception("invalid structure %s" % structure)

    # structures are either strings or tuples
    assert(isinstance(structure, tuple))
    
    if structure[0] == 'lowrank':
        model1 = build_model(structure[1], (N, K), settings, tree_path + "l")
        model2 = build_model(structure[2], (D, K), settings, tree_path + "r")
        model = build_lowrank(model1, model2, settings, tree_path)
    elif structure[0] == 'cluster':
        centers = build_model(structure[1],  (K, D), settings, tree_path + "c")
        model = build_clustering(centers, (N, D), settings, tree_path)
    elif structure[0] == 'features':
        b = build_model(structure[1], (N, K), settings, tree_path + "b")
        g = build_model(structure[2], (K, D), settings, tree_path + "f")
        model = build_features(b, g, settings, tree_path)
    elif structure[0] == 'chain':
        g = build_model(structure[1], (N, D), settings, tree_path + "C")
        model = build_chain(g, settings, tree_path)
    elif structure[0] == "sparse":
        g = build_model(structure[1], (N, D), settings, tree_path + "s")
        model = build_sparsity(g, settings, tree_path)
    elif structure[0] == "transpose":
        g = build_model(structure[1], (D, N), settings, tree_path + "t")
        model = build_transpose(g, tree_path)
    else:
        raise Exception("invalid structure %s" % repr(structure))

    return model


class ExperimentSettings(object):

    def __init__(self):
        self.gaussian_ard = True
        self.gaussian_colprec_alpha = np.float32(2.0)
        self.gaussian_colprec_beta = np.float32(0.5)
        self.noise_prec_alpha = np.float32(1.0)
        self.noise_prec_beta = np.float32(0.01)

        self.beta_prior_params = np.float32(1.0), np.float32(1.0)

        self.constant_noise_std = None
        self.dirichlet_alpha = np.float32(1.0)
        self.max_rank = 6
        
def main():
    from bayesflow.models.train import construct_elbo, optimize_elbo, print_inference_summary
    
    N = 100
    D = 30

    X = np.float32(np.random.randn(N, D))
    
    settings = ExperimentSettings()
    
    structures = list(list_structures(1))[5:6]
    for structure in structures:
        m = build_model(structure, (N, D), settings)
        print "built model for structure", repr(structure)
        print "model is", m
        print
        m.observe(X)

        elbo_terms, posterior = optimize_elbo(m)
        print_inference_summary(elbo_terms, posterior)
        
if __name__ == "__main__":
    main()
