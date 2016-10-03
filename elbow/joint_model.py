import numpy as np
import tensorflow as tf

import uuid
import copy
import time

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

    def __init__(self, *nodes, **kwargs):

        # want to define this constructor as __init__(self, *nodes,
        # named_arg1=default1, etc) but python 2 doesn't allow named
        # arguments after optional positional arguments, so we have
        # to fake it by parsing kwargs manually.
        args = {'minibatch_ratio': 1.0}
        for arg, default_val in args.items():
            if arg in kwargs:
                args[arg] = kwargs[arg]
                del kwargs[arg]
        if len(kwargs) > 0:
            raise TypeError('unexpected keyword arguments %s' % (kwargs.keys()))
        self.__dict__.update(args)
        
        # the model includes all ancestors of the passed-in nodes. 
        self.component_nodes = ancestor_closure(nodes)
        self.by_name = {n.name : n for n in self.component_nodes}

        # don't compute the variational nodes until actually needed by the ELBO constructor.
        # this allows us to attach variational nodes *after* creating the joint model, which
        # is useful if we want to observe sample from the joint model. 
        self.variational_nodes = None

        self.session = None
        self.feeder = None
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

            vnodes = self.get_variational_nodes()
            
            global_elps = [n.expected_logp() for n in self.component_nodes if not n.local]
            local_elps = [n.expected_logp() for n in self.component_nodes if n.local]

            global_entropies = [n.entropy() for n in vnodes if not n.local]
            local_entropies = [n.entropy() for n in vnodes if n.local]

            symmetry_correction = tf.reduce_sum(tf.pack([n._hack_symmetry_correction() for n in self.component_nodes]))
            
            elp = tf.reduce_sum(tf.pack(global_elps)) + self.minibatch_ratio * tf.reduce_sum(tf.pack(local_elps))
            entropy = tf.reduce_sum(tf.pack(global_entropies)) + self.minibatch_ratio * tf.reduce_sum(tf.pack(global_entropies))
            
            self.elbo = elp+entropy + symmetry_correction
            
        return self.elbo

    #def build_variational_model(self):
    #    explicit_qnodes = [node.q_distribution() for node in self.component_nodes]
    #    return [node for node in ancestor_closure(explicit_qnodes) if node not in self.component_nodes]

    def build_variational_model(self):
        # start with nodes that already have attached Q distributions
        attached = [n for n in self.component_nodes if n._q_distribution is not None]

        i = 0
        while i < len(attached):
            n = attached[i]
            networks = n.inference_networks()
            for inp_name, q in networks.items():
                if inp_name in n.inputs_random:
                    inp_node = n.inputs_random[inp_name]
                    if not inp_node.local:
                        print "WARNING: attaching inference network to node %s not marked as local" % (inp_node)
                        
                    try:
                        inp_node.attach_q(q)
                        print "attached inference network from  %s to input %s (%s)" % (n, inp_name, inp_node)
                        attached.append(inp_node)
                    except Exception as e:
                        import pdb; pdb.set_trace()
                        print "WARNING: cannot attach inference network from %s to input %s (%s): %s" % (n, inp_name, inp_node, e)
            i += 1

        for node in self.component_nodes:
            if node.local and (node._q_distribution is None):
                raise Exception("node %s marked as local but no inference network is attached!" % (node))
            
        explicit_qnodes = [node.q_distribution() for node in self.component_nodes]
        return [node for node in ancestor_closure(explicit_qnodes) if node not in self.component_nodes]

    def get_variational_nodes(self):
        if self.variational_nodes is None:
            self.variational_nodes = self.build_variational_model()
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
        vals = sess.run(all_terms, feed_dict=self.feed_dict())

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
                try:
                    d[name] = session.run(inp, feed_dict=self.feed_dict())
                except Exception as e: # from fetching a placeholder
                    print e
                    continue
            if len(d) > 0:
                posterior_vals[node.name] = d
        return posterior_vals

    def sample(self, seed=0):
        sess = self.get_session(seed=seed)
        samples = {}

        sampled = sess.run([node._sampled for node in self.component_nodes], feed_dict=self.feed_dict())
        
        for node, sval in zip(self.component_nodes, sampled):
            samples[node.name] = sval
        return samples
        
    def train(self, adam_rate=0.1, stopping_rule=None, steps=None,
              avg_decay=None, debug=False, print_s=1):
        elbo = self.construct_elbo()


        if stopping_rule is None:
            if steps is not None:
                stopping_rule = StepCountStopper(step_count=steps)
            elif avg_decay is not None:
                stopping_rule = MovingAverageStopper(decay=avg_decay)
            else:
                stopping_rule = MovingAverageStopper()
        try:
            train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
        except ValueError as e:
            print e
            return
            
        if debug:
            debug_ops = tf.add_check_numerics_ops()

        session = self.get_session(do_init=False)

        init = tf.initialize_all_variables()
        session.run(init)
        
        elbo_val = None
        running_elbo = 0
        i = 0
        t = time.time()
        stopping_rule.reset()
        while not stopping_rule.observe(elbo_val):
            if debug:
                session.run(debug_ops)

            fd = self.feed_dict()
                
            session.run(train_step, feed_dict=fd)

            elbo_val = session.run((elbo), feed_dict=fd)
            if print_s is not None and (time.time() - t) > print_s:
                print i, elbo_val
                t = time.time()
                
            i += 1
            

    def feed_dict(self):
        if self.feeder is not None:
            return self.feeder()
        else:
            return None

    def register_feed(self, feeder):
        self.feeder = feeder
        
    def monte_carlo_elbo(self, n_samples):
        
        sess = self.get_session()
        elbo = self.construct_elbo()
        samples = [sess.run(elbo, feed_dict=self.feed_dict()) for i in range(n_samples)]
        return np.mean(samples)

    def __del__(self):
        if self.session is not None:
            self.session.close()

class StepCountStopper(object):

    def __init__(self, step_count=1000):
        self.step_count = step_count
        self.steps = 0

    def reset(self):
        self.steps = 0
        
    def observe(self, v):
        self.steps += 1
        if v is not None and not np.isfinite(v):
            return True

        return self.steps > self.step_count
        
class MovingAverageStopper(object):

    def __init__(self, decay=0.98, eps=0.5, min_steps = 10):
        self.decay = decay
        self.eps=eps
        self.min_steps = min_steps

    def reset(self):
        self.steps = 0
        self.moving_average = None
        
    def observe(self, v):
        self.steps += 1

        if v is None:
            return False
        
        if not np.isfinite(v):
            return True
        
        if self.moving_average is None:
            self.moving_average = v
            return False
        
        old_avg = self.moving_average
        self.moving_average *= self.decay
        self.moving_average += (1-self.decay)*v

        return (self.steps > self.min_steps and self.moving_average <= old_avg + self.eps)


class BatchGenerator(object):
    """
    Return random sets of batch_size rows from an input numpy array. 
    """
    
    def __init__(self, data, batch_size):
        self.data = data
        self.n = data.shape[0]
        self.batch_size=batch_size

        self.idx = self.n+1

    def next_batch(self):
        if self.idx + self.batch_size > self.n:
            self.idx = 0
            self.shuffled = self.data[np.random.permutation(self.n)]

        batch = self.shuffled[self.idx:self.idx+self.batch_size]
        self.idx += self.batch_size
        return batch
                                                                                                                            
