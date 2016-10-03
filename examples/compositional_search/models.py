import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.elementary import Gaussian, BernoulliMatrix, GammaMatrix, BetaMatrix, DirichletMatrix
from bayesflow.joint_model import Model, BatchGenerator
from bayesflow.models.factorizations import *
from bayesflow.transforms import DeterministicTransform, TransformedDistribution, Exp
import bayesflow.transforms as transforms

from grammar import list_structures

def build_column_stds(shape, settings, name):
    N, K = shape

    if settings.gaussian_auto_ard:
        prec_dim = K


        logstd_mean = Gaussian(mean=0.0, std=1.0, shape=(1,),
                                     dtype=np.float32, name="%s_logstd_mean" % name)
        logstd_logstd = Gaussian(mean=0.0, std=1.0, shape=(1,),
                                       dtype=np.float32, name="%s_logstd_logstd" % name)
        logstd_std = DeterministicTransform(logstd_logstd, transforms.Exp,
                                            name="%s_logstd_std" % name)


        # model column stds as drawn from a lognormal distribution with inferred mean and std.
        # this allows for ARD if we infer a high variance on the column stds, but cheaply
        # specializes to the case where all column variances are the same
        logstd = Gaussian(mean=logstd_mean, std=logstd_std, shape=(prec_dim,),
                                dtype=np.float32, name="%s_logstd" % name)
        std = DeterministicTransform(logstd, transforms.Exp,
                                     name="%s_std" % name)
    else:
        std = settings.constant_gaussian_std


    return std
    
def build_gaussian(shape, settings, name, local=False):
    N, K = shape

    col_stds = build_column_stds(shape, settings, name)
    
    G = Gaussian(mean=0.0, std=col_stds, name=name, shape=shape, local=local)
    
    return G

def build_bernoulli(shape, settings, name, local=False):
    N, K = shape
    a, b = settings.beta_prior_params
    
    pi = BetaMatrix(alpha=a, beta=b, shape=(K,), name="%s_pi" % name)
    B = BernoulliMatrix(p=pi, shape=(N, K), name=name, local=local)

    return B

def build_transpose(A, name):
    return DeterministicTransform(A, transforms.Transpose, name=name)

def build_noise_std(settings, name):
    if settings.constant_noise_std is not None:
        noisestd = settings.constant_noise_std
    else:
        alpha = settings.noise_prec_alpha
        beta = settings.noise_prec_beta
        noiseprec = GammaMatrix(alpha=alpha, beta=beta, shape=(1,),
                                dtype=np.float32, name="%s_noise_precs" % name)
        noisestd = DeterministicTransform(noiseprec, transforms.Reciprocal_Sqrt,
                                          name="%s_noise_std" % name)
                
    return noisestd

def build_lowrank(G1, G2, settings, name, local=False):
    noise_std = build_noise_std(settings, name)

    #if not isinstance(G1, DeterministicTransform):
    #    G1.attach_q(Gaussian(shape=G1.shape))
    #if not isinstance(G2, DeterministicTransform):
    #    G2.attach_q(Gaussian(shape=G2.shape))
    
    return NoisyGaussianMatrixProduct(G1, G2, noise_std, name=name, local=local)

def build_features(B, G, settings, name, local=False):
    noise_std = build_noise_std(settings, name)
    #if not isinstance(G, DeterministicTransform):
    #    G.attach_q(Gaussian(shape=G.shape))

    return NoisyLatentFeatures(B, G, noise_std, name=name, local=local)

def build_chain(G, settings, name, local=False):
    noise_std = build_noise_std(settings, name)

    #if not isinstance(G, DeterministicTransform):
    #    G.attach_q(Gaussian(shape=G.shape))

    C = NoisyCumulativeSum(G, noise_std, name=name, local=local)
    
    return C

def build_sparsity(G, settings, name, local=False):
    #if not isinstance(G, DeterministicTransform):
    #    G.attach_q(Gaussian(shape=G.shape))

    expG = DeterministicTransform(G, Exp, name="%s_exp" % name)
    stds = build_column_stds(G.shape, settings, name)
    return MultiplicativeGaussianNoise(expG, stds, name=name, local=local)

def build_clustering(centers, shape, settings, name, local=False):
    N, D = shape
    K, D2 = centers.shape
    assert(D==D2)

    #if not isinstance(centers, DeterministicTransform):
    #    centers.attach_q(Gaussian(shape=centers.shape))

    weights = DirichletMatrix(alpha=settings.dirichlet_alpha,
                              shape=(K,),
                              name="%s_weights" % name)


    noise_std = build_noise_std(settings, name)
    return GMMClustering(weights=weights, centers=centers,
                         std=noise_std, shape=shape, name=name, local=local)

def build_model(structure, shape, settings, tree_path="g", local=True):

    N, D = shape
    K = settings.max_rank
    
    if isinstance(structure, str):
        if structure == "g":
            return build_gaussian(shape, settings, name="%s_g" % tree_path, local=local)
        elif structure == 'b':
            return build_bernoulli(shape, settings, name="%s_b" % tree_path, local=local)
        else:
            raise Exception("invalid structure %s" % structure)

    # structures are either strings or tuples
    assert(isinstance(structure, tuple))
    
    if structure[0] == 'lowrank':
        model1 = build_model(structure[1], (N, K), settings, tree_path + "l", local=local)
        model2 = build_model(structure[2], (D, K), settings, tree_path + "r", local=False)
        model = build_lowrank(model1, model2, settings, tree_path, local=local)
    elif structure[0] == 'cluster':
        centers = build_model(structure[1],  (K, D), settings, tree_path + "c", local=False)
        model = build_clustering(centers, (N, D), settings, tree_path, local=local)
    elif structure[0] == 'features':
        b = build_model(structure[1], (N, K), settings, tree_path + "b", local=local)
        g = build_model(structure[2], (K, D), settings, tree_path + "f", local=False)
        model = build_features(b, g, settings, tree_path, local=local)
    elif structure[0] == 'chain':
        g = build_model(structure[1], (N, D), settings, tree_path + "C")
        model = build_chain(g, settings, tree_path)
    elif structure[0] == "sparse":
        g = build_model(structure[1], (N, D), settings, tree_path + "s", local=local)
        model = build_sparsity(g, settings, tree_path, local=local)
    elif structure[0] == "transpose":
        g = build_model(structure[1], (D, N), settings, tree_path + "t")
        model = build_transpose(g, tree_path)
    else:
        raise Exception("invalid structure %s" % repr(structure))

    return model


def sanity_check_model_recovery(structures, settings):

    N = 10000
    D = 100
    
    samples = [build_model(structure, (N, D), settings).sample() for structure in structures]
        
    for i, sample in enumerate(samples):
        print "using X sampled from", structures[i]

        batch_N = 32
        
        scores = []
        for structure in structures:
            m = build_model(structure, (batch_N, D), settings, local=True)
            jm = Model()
            print "built model for structure", repr(structure)
            print "model is", m
            print
            obsM = m.observe_placeholder()

            jm = Model(m, minibatch_ratio = N/float(batch_N))
            b = BatchGenerator(sample, batch_size=batch_N)
            jm.register_feed(lambda : {obsM : b.next_batch()})

            jm.train(stopping_rule=settings.stopping_rule,
                     adam_rate=settings.adam_rate)
            score = jm.monte_carlo_elbo(n_samples=settings.n_elbo_samples)
            scores.append((score))

        print "results for sample from", structures[i]
        for structure, score in zip(structures, scores):
            print structure, score
        best_structure = structures[np.argmax(scores)]
        print "best structure", best_structure
        if best_structure != structures[i]:
            print "WARNING DOES NOT MATCH TRUE STRUCTURE"
    
def main():
    from bayesflow import Model
    from search import ExperimentSettings
    
    N = 100
    D = 10

    settings = ExperimentSettings()    
    structures = list(list_structures(1))

    settings.max_rank=2
    settings.gaussian_auto_ard = False
    settings.constant_gaussian_std = 1.0
    settings.constant_noise_std = 0.1

    structures = [('features', 'b', 'g'), ('features', 'b', 'g'), ('features', 'b', 'g'), ('features', 'b', 'g'), ('lowrank', 'g', 'g'), ('g'), ('cluster', 'g')]
    
    sanity_check_model_recovery(structures, settings)
    
if __name__ == "__main__":
    main()
