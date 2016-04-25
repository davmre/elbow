import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import FlatDistribution
from bayesflow.models.elementary import GaussianMatrix, BernoulliMatrix, GammaMatrix, BetaMatrix, DirichletMatrix
from bayesflow.models.q_distributions import DeltaQDistribution, GaussianQDistribution, BernoulliQDistribution, SimplexQDistribution
from bayesflow.models.matrix_decompositions import *
from bayesflow.models.transforms import PointwiseTransformedMatrix, PointwiseTransformedQDistribution, Transpose


from grammar import list_structures

def build_column_stds(shape, settings, name):
    N, K = shape

    if settings.gaussian_auto_ard:
        prec_dim = K


        logstd_mean = GaussianMatrix(mean=0.0, std=1.0, output_shape=(1,),
                                     dtype=np.float32, name="%s_logstd_mean" % name)
        logstd_logstd = GaussianMatrix(mean=0.0, std=1.0, output_shape=(1,),
                                       dtype=np.float32, name="%s_logstd_logstd" % name)
        logstd_std = PointwiseTransformedMatrix(logstd_logstd, bf.transforms.exp,
                                                name="%s_logstd_std" % name)


        # model column stds as drawn from a lognormal distribution with inferred mean and std.
        # this allows for ARD if we infer a high variance on the column stds, but cheaply
        # specializes to the case where all column variances are the same
        logstd = GaussianMatrix(mean=logstd_mean, std=logstd_std, output_shape=(prec_dim,),
                                dtype=np.float32, name="%s_logstd" % name)
        std = PointwiseTransformedMatrix(logstd, bf.transforms.exp,
                                         name="%s_std" % name)
    else:
        std = settings.constant_gaussian_std

        #prec = GammaMatrix(alpha=alpha, beta=beta, output_shape=(prec_dim,),
    #                        dtype=np.float32, name="%s_precs" % name)
    #std = PointwiseTransformedMatrix(prec, bf.transforms.reciprocal_sqrt,
    #                                 name="%s_std" % name)

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
    B = BernoulliMatrix(p=pi, output_shape=(N, K), name=name)

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
                
    return noisestd

def build_lowrank(G1, G2, settings, name):
    noise_std = build_noise_std(settings, name)

    G1.attach_q(GaussianQDistribution(G1.output_shape))
    G2.attach_q(GaussianQDistribution(G2.output_shape))
    
    return NoisyGaussianMatrixProduct(G1, G2, noise_std, name=name)

def build_features(B, G, settings, name):
    noise_std = build_noise_std(settings, name)
    G.attach_q(GaussianQDistribution(G.output_shape))

    return NoisyLatentFeatures(B, G, noise_std, name=name)

def build_chain(G, settings, name):
    noise_std = build_noise_std(settings, name)
    G.attach_q(GaussianQDistribution(G.output_shape))    

    return NoisyCumulativeSum(G, noise_std, name=name)

def build_sparsity(G, settings, name):
    G.attach_q(GaussianQDistribution(G.output_shape))    
    expG = PointwiseTransformedMatrix(G, bf.transforms.exp, name="%s_exp" % name)
    stds = build_column_stds(G.output_shape, settings, name)
    return MultiplicativeGaussianNoise(expG, stds, name=name)

def build_clustering(centers, shape, settings, name):
    N, D = shape
    K, D2 = centers.output_shape
    assert(D==D2)

    centers.attach_q(GaussianQDistribution(centers.output_shape))

    weights = DirichletMatrix(alpha=settings.dirichlet_alpha,
                              output_shape=(K,),
                              name="%s_weights" % name)


    noise_std = build_noise_std(settings, name)
    return GMMClustering(weights=weights, centers=centers,
                         std=noise_std, output_shape=shape, name=name)

def build_model(structure, shape, settings, tree_path="g"):

    N, D = shape
    K = settings.max_rank
    
    if isinstance(structure, str):
        if structure == "g":
            return build_gaussian(shape, settings, name="%s_g" % tree_path)
        elif structure == 'b':
            return build_bernoulli(shape, settings, name="%s_b" % tree_path)
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


        
def main():
    from bayesflow.models.train import construct_elbo, optimize_elbo, print_inference_summary
    
    N = 50
    D = 50

    X = np.float32(np.random.randn(N, D))
    
    settings = ExperimentSettings()
    
    structures = list(list_structures(1))
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
