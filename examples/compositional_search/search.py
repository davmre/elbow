import numpy as np
import tensorflow as tf
import copy

import bayesflow as bf
import bayesflow.util as util

from bayesflow.joint_model import Model

from grammar import list_successors
from models import build_model


class ExperimentSettings(object):

    def __init__(self):
        self.gaussian_auto_ard = True
        self.constant_gaussian_std = None

        self.noise_prec_alpha = np.float32(1.0)
        self.noise_prec_beta = np.float32(0.01)

        self.beta_prior_params = np.float32(1.0), np.float32(1.0)

        self.constant_noise_std = None
        self.dirichlet_alpha = np.float32(1.0)
        self.max_rank = 6

        self.adam_rate = 0.1
        self.steps = 500
        self.beamsize = 2
        self.p_stop_structure = 0.3
        self.n_elbo_samples = 50
        
def beamsearch_helper(beam, beamsize=2):
    scores = [score(model) for model in beam]
    perm = sorted(np.arange(len(beam)), key = lambda i : -scores[i])
    sorted_beam = beam[perm]
    best = sorted_beam[:beamsize]

    newbeam = []
    for oldmodel in best:
        successors = list_successors(oldmodel)
        newbeam += [oldmodel,] + successors
    return newbeam


def score_model(structure, X, settings):
    N, D = X.shape
    m = build_model(structure, (N, D), settings)
    m.observe(X)

    elbo, sample_stochastic, decompose_elbo, inspect_posterior = construct_elbo(m)

    steps = settings.steps
    try:
        train_step = tf.train.AdamOptimizer(settings.adam_rate).minimize(-elbo)
    except:
        steps = 0
        
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(steps):
        fd = sample_stochastic()
        sess.run(train_step, feed_dict = fd)
        
    score = monte_carlo_elbo(elbo, sess, sample_stochastic, n_samples=settings.n_elbo_samples)

    sess.close()
    
    return score

def initialize_from(old_model, new_model):
    for name, new_node in new_model.input_nodes.items():
        if name in old_model.input_nodes:
            old_node = old_model.input_nodes[name]
            initialize_from(old_node, new_node)
    
    old_qdist = old_model.q_distribution()
    new_qdist = old_model.q_distribution()
    if isinstance(old_qdist, GaussianQistribution):
        # TODO figure out the right, elegant way to do this
        # see doc for discussion
        # new_qdist._intialize(old_qdist.mean, old_qdist.logodds)
        pass
def score_and_sort_beam(beam, X, settings):
    scored_beam = []
    for (structure, structure_logp, model_score) in beam:
        score = score_model(structure, X, settings) + structure_logp
        print "score", score, "for structure", structure
        scored_beam.append((structure, structure_logp, score))
    sorted_beam = sorted(scored_beam, key = lambda a : -a[2])
    return sorted_beam

def expand_beam(beam, settings):
    continue_logp = np.log(1.0-settings.p_stop_structure)
    new_beam = copy.copy(beam)
    for (structure, structure_score, model_score) in beam:
        successors = list_successors(structure)
        for successor in successors:
            new_score = structure_score + continue_logp - np.log(len(successors))
            new_beam.append((successor, new_score, None))
    return new_beam
            
def do_structure_search(X, settings):

    base_logprob = np.log(settings.p_stop_structure)
    structure_beam = [('g', base_logprob, 0.0),]
    best_structures = score_and_sort_beam(structure_beam, X, settings)
    old_best_score = -np.inf
    best_score = best_structures[0][2]

    i = 0
    while best_score > old_best_score:
        structure_beam = expand_beam(best_structures, settings)
        scored_structures = score_and_sort_beam(structure_beam, X, settings)
        best_structures = scored_structures[:settings.beamsize]
        old_best_score = best_score
        best_score = best_structures[0][2]

        i+=1
        print "epoch %d" % i, "beam", best_structures
    
def main():
    N = 50
    D = 20

    settings = ExperimentSettings()
    settings.max_rank=2
    settings.gaussian_auto_ard = False
    settings.constant_gaussian_std = 1.0
    settings.constant_noise_std = 0.1
    
    #X = np.float32(np.random.randn(N, D))

    m = build_model(('lowrank', ('chain', 'g'), 'g'), (N, D), settings)
    #m = build_model(('chain', 'g'), (N, D), settings)
    X = m.sample()
    #X /= np.std(X)
    


    best_structure = do_structure_search(X, settings)

    
if __name__ == "__main__":
    main()
