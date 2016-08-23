import numpy as np
import tensorflow as tf

import uuid
import copy

from bayesflow.models.transforms import DeterministicTransform
from bayesflow.models import ConditionalDistribution, WrapperNode

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

        # the variational model includes all ancestors of qdists associated
        # with model nodes.
        # WARNING this is only correct if we stop at the first stochastic
        # ancestor (of deterministic qdists). If there are latent stochastic
        # variables in the variational model we need the HVM objective
        # which is not yet implemented. 
        explicit_qnodes = [node.q_distribution() for node in self.component_nodes]
        self.variational_nodes = ancestor_closure(explicit_qnodes)
                
    def construct_elbo(self):
        elps = [n.expected_logp() for n in self.component_nodes]
        entropies = [n.entropy() for n in self.variational_nodes]
        
        elp = tf.reduce_sum(tf.pack(elps))
        entropy = tf.reduce_sum(tf.pack(entropies))
        elbo = elp+entropy
        return elbo
        
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
        
    def train(self, steps=200, adam_rate=0.1, debug=False, return_session=False):
        elbo = self.construct_elbo()

        try:
            train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
        except ValueError as e:
            print e
            steps = 0

        init = tf.initialize_all_variables()

        if debug:
            debug_ops = tf.add_check_numerics_ops()

        sess = tf.Session()
        sess.run(init)
        for i in range(steps):
            if debug:
                sess.run(debug_ops)

            sess.run(train_step)

            elbo_val = sess.run((elbo))
            print i, elbo_val

        posterior = self.posterior(session=sess)
            
        return posterior
    
