import numpy as np
import tensorflow as tf

import uuid
import copy

from conditional_dist import ConditionalDistribution, WrapperNode
from transforms import DeterministicTransform

def ancestors(node):
    return set([node,] + [ancestor_node for inp in node.inputs_random.values() for ancestor_node in ancestors(inp)])

def ancestor_closure(nodes):
    all_ancestors = set()
    for node in nodes:
        all_ancestors = all_ancestors.union(ancestors(node))
    return all_ancestors

class Model(object):

    def __init__(self, *nodes):

        # the model includes all ancestors of the passed-in nodes. 
        self.component_nodes = ancestor_closure(nodes)
        self.by_name = {n.name : n for n in self.component_nodes}

        # don't compute the variational nodes until actually needed by the ELBO constructor.
        # this allows us to attach variational nodes *after* creating the joint model, which
        # is useful if we want to observe sample from the joint model. 
        self.variational_nodes = None

        self.session = None
        self.elbo = None
        
    def __getitem__(self, a):
        return self.by_name[a]

    def construct_elbo(self):

        if self.elbo is None:
        
            # the variational model includes all ancestors of qdists associated
            # with model nodes.
            # WARNING this is only correct if we stop at the first stochastic
            # ancestor (of deterministic qdists). If there are latent stochastic
            # variables in the variational model we need the HVM objective
            # which is not yet implemented. 

            elps = [n.expected_logp() for n in self.component_nodes]
            entropies = [n.entropy() for n in self.get_variational_nodes()]

            elp = tf.reduce_sum(tf.pack(elps))
            entropy = tf.reduce_sum(tf.pack(entropies))
            self.elbo = elp+entropy
            
        return self.elbo

    def get_variational_nodes(self):
        if self.variational_nodes is None:
            explicit_qnodes = [node.q_distribution() for node in self.component_nodes]
            self.variational_nodes = [node for node in ancestor_closure(explicit_qnodes) if node not in self.component_nodes]
        return self.variational_nodes
    
    def elbo_terms(self):
        elps = {n.name: n.expected_logp() for n in self.component_nodes}
        entropies = {n.name: n.entropy() for n in self.get_variational_nodes()}
        return elps, entropies

    def get_session(self, seed=0, do_init=True):

        if self.session is None:
            tf.set_random_seed(seed)
            self.session = tf.Session()

            if do_init:
                init = tf.initialize_all_variables()
                self.session.run(init)
            
        return self.session
    
    def evaluate_elbo_terms(self):

        sess = self.get_session()
        
        elps, entropies = self.elbo_terms()

        sorted_elps = sorted(elps.items())
        sorted_entropies = sorted(entropies.items())
        elp_names, elp_terms = zip(*sorted(elps.items()))
        entropy_names, entropy_terms = zip(*sorted(entropies.items()))
        all_terms = elp_terms + entropy_terms
        vals = sess.run(all_terms, feed_dict)

        n_elp = len(elp_terms)
        elp_vals = dict(zip(elp_names, vals[:n_elp]))
        entropy_vals = dict(zip(entropy_names, vals[n_elp:]))

        return elp_vals, entropy_vals
        
    def posterior(self):
        session = self.get_session()
        posterior_vals = {}
        for node in self.variational_nodes:
            d = {}
            for name, inp in node.inputs_nonrandom.items():                
                d[name] = session.run(inp)
            if len(d) > 0:
                posterior_vals[node.name] = d
        return posterior_vals

    def sample(self, seed=0):
        sess = self.get_session(seed=seed)
        samples = {}

        sampled = sess.run([node._sampled for node in self.component_nodes])
        
        for node, sval in zip(self.component_nodes, sampled):
            samples[node] = sval
        return samples
        
    def train(self, eps=1e-6, steps=1000, adam_rate=0.1, debug=False):
        elbo = self.construct_elbo()

        try:
            train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
        except ValueError as e:
            print e
            steps = 0

        if debug:
            debug_ops = tf.add_check_numerics_ops()

        session = self.get_session(do_init=False)

        init = tf.initialize_all_variables()
        session.run(init)
        
        old_elbo_val = -np.inf
        for i in range(steps):
            if debug:
                session.run(debug_ops)

            session.run(train_step)

            elbo_val = session.run((elbo))
            print i, elbo_val
            if eps is not None and np.abs(elbo_val - old_elbo_val) < eps:
                break
            else:
                old_elbo_val = elbo_val
            
    def monte_carlo_elbo(self, n_samples):
        
        sess = self.get_session()
        elbo = self.construct_elbo()
        samples = [sess.run(elbo) for i in range(n_samples)]
        return np.mean(samples)

    def __del__(self):
        if self.session is not None:
            self.session.close()
