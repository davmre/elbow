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

    def construct_elbo(self):

        # the variational model includes all ancestors of qdists associated
        # with model nodes.
        # WARNING this is only correct if we stop at the first stochastic
        # ancestor (of deterministic qdists). If there are latent stochastic
        # variables in the variational model we need the HVM objective
        # which is not yet implemented. 
        explicit_qnodes = [node.q_distribution() for node in self.component_nodes]
        self.variational_nodes = [node for node in ancestor_closure(explicit_qnodes) if node not in self.component_nodes]
        
        elps = [n.expected_logp() for n in self.component_nodes]
        entropies = [n.entropy() for n in self.variational_nodes]
        
        elp = tf.reduce_sum(tf.pack(elps))
        entropy = tf.reduce_sum(tf.pack(entropies))
        elbo = elp+entropy
        return elbo

    def elbo_terms(self):
        elps = {n.name: n.expected_logp() for n in self.component_nodes}
        entropies = {n.name: n.entropy() for n in self.variational_nodes}
        return elps, entropies

    def evaluate_elbo_terms(self, sess):
        elps, entropies = self.elbo_terms()

        sorted_elps = sorted(elps.items())
        sorted_entropies = sorted(entropies.items())
        elp_names, elp_terms = zip(*sorted(elps.items()))
        entropy_names, entropy_terms = zip(*sorted(entropies.items()))
        all_terms = elp_terms + entropy_terms
        vals = sess.run(all_terms)

        n_elp = len(elp_terms)
        elp_vals = dict(zip(elp_names, vals[:n_elp]))
        entropy_vals = dict(zip(entropy_names, vals[n_elp:]))

        return elp_vals, entropy_vals
        
    def posterior(self, session, feed_dict=None):
        posterior_vals = {}
        for node in self.variational_nodes:

            if isinstance(node, WrapperNode):
                continue
            
            d = {}
            for name, inp in node.inputs_nonrandom.items():
                d[name] = session.run(inp)
            if len(d) > 0:
                posterior_vals[node.name] = d
        return posterior_vals

    def sample(self, seed=0):
        tf.set_random_seed(seed)
        
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        samples = {}

        sampled = sess.run([node._sampled for node in self.component_nodes])
        
        for node, sval in zip(self.component_nodes, sampled):
            samples[node] = sval
        return samples
        
    def train(self, steps=200, adam_rate=0.1, debug=False, session=None):
        elbo = self.construct_elbo()

        try:
            train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
        except ValueError as e:
            print e
            steps = 0

        init = tf.initialize_all_variables()

        if debug:
            debug_ops = tf.add_check_numerics_ops()

        session = tf.Session() if session is None else session
        session.run(init)
        for i in range(steps):
            if debug:
                session.run(debug_ops)

            session.run(train_step)

            elbo_val = session.run((elbo))
            print i, elbo_val

        posterior = self.posterior(session)
        return posterior

    
