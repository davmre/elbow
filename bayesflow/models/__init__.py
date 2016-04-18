import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models.q_distributions import ObservedQDistribution, GaussianQDistribution

class ConditionalDistribution(object):
    """
    
    Generic object representing a conditional distribution: a single output conditioned on some set of inputs. 
    Unconditional distributions are the special case where the inputs are the empty set. 

    The inputs are assumed to be random variables described by their own respective distributions. 
    Thus the object graph implicitly represents a directed graphical model (Bayesian network).

    """

    
    def __init__(self, output_shape=None, dtype=None, name=None, **kwargs):
        # store map of input param names to the nodes modeling those params
        self.input_nodes = {}
        for input_name in self.inputs():
            # inputs can be provided as constants, or nodes modeled by bayesflow distributions.
            # if they are constants, we create a node to represent that. 
            if isinstance(kwargs[input_name], ConditionalDistribution):
                self.input_nodes[input_name] = kwargs[input_name]
            else:
                constant_node = FlatDistribution(kwargs[input_name], fixed=True)
                self.input_nodes[input_name] = constant_node

        self.name = name
                
        # compute the shape of the output at this node
        if output_shape is not None:
            self.output_shape = output_shape
        else:
            input_shapes = {name + "_shape": node.output_shape for (name,node) in self.input_nodes.items()}
            self.output_shape = self._compute_shape(**input_shapes)

        if dtype is not None:
            self.dtype = dtype
        else:
            input_dtypes = {name + "_dtype": node.dtype for (name,node) in self.input_nodes.items()}
            self.dtype = self._compute_dtype(**input_dtypes)
        
        # compute the list of all ancestor nodes in the graph, by merging the ancestor lists of the parent nodes. 
        # Storing this for every node is slightly inefficient, but makes for simpler code by avoiding the need for a
        # graph traversal when constructing joint quantities like the ELBO. This could be optimized if the ancestor lists
        # ever become a performance bottleneck. 
        self.ancestors = set( [self,] + [ancestor for node in self.input_nodes.values() for ancestor in node.ancestors ] )
    
        self._sampled_value = None
        self._sampled_value_seed = None
            
    def sample(self, seed=0):
        if seed != self._sampled_value_seed:
            input_samples = {name: node.sample(seed=seed) for (name,node) in self.input_nodes.items()}        
            
            np.random.seed(seed)
            self._sampled_value_seed = seed
            self._sampled_value = self._sample(**input_samples)
        
        return self._sampled_value
    
    def attach_q(self, q_distribution):
        # TODO check that the types and shape of the Q distribution match
        self.q_distribution = q_distribution
    
    def observe(self, observed_val):
        qdist = ObservedQDistribution(observed_val)
        self.attach_q(qdist)
        return qdist
        
    def attach_gaussian_q(self, **kwargs):
        q = GaussianQDistribution(shape=self.output_shape, **kwargs)
        self.attach_q(q)
        return q
    
    def elbo_term(self):
        input_qs = {"q_"+name: node.q_distribution for (name,node) in self.input_nodes.items()}
        expected_lp = self._expected_logp(q_result = self.q_distribution, **input_qs)
        entropy = self.q_distribution.entropy()
        return expected_lp, entropy

    def _expected_logp(self, **kwargs):
        # default implementation: compute E_q[ log p(x) ] as a Monte Carlo sample.
        samples = {}
        for (q_key, qdist) in kwargs.items():
            assert(q_key.startswith("q_"))
            key = q_key[2:]
            samples[key] = qdist.sample
        return self._logp(**samples)
    
class FlatDistribution(ConditionalDistribution):

    """
    A "dummy" distribution object representing a flat prior. It requires a "default" value that will
    be returned when sampling from this distribution. 
    
    Parameters with known or fixed values can be represented using a flat prior and an ObservedQDistribution 
    to fix them to the given value. 
    """
    
    def __init__(self, value, fixed=True):
        try:
            self.dtype = value.dtype
        except:
            self.dtype = np.float32
        
        self.value = np.asarray(value, dtype=self.dtype) 
        output_shape = self.value.shape        
        super(FlatDistribution, self).__init__(output_shape = output_shape)

        if fixed:
            qdist = ObservedQDistribution(self.value)
            self.attach_q(qdist)
        
    def inputs(self):
        return ()
        
    def _sample(self, seed=0):
        return self.value
    
    def _compute_dtype(self):
        return self.dtype
        
    def _expected_logp(self, q_result):        
        return 0.0


    
def construct_elbo(*evidence_nodes):
    model_nodes = set([ancestor for node in evidence_nodes for ancestor in node.ancestors])
    expected_likelihoods, entropies = zip(*[node.elbo_term() for node in model_nodes])
    expected_likelihood = tf.reduce_sum(tf.pack(expected_likelihoods))
    entropy = tf.reduce_sum(tf.pack(entropies))
    
    q_distributions = set([node.q_distribution for node in model_nodes])
    def sample_stochastic_inputs():
        return {var: val for qdist in q_distributions for (var, val) in qdist.sample_stochastic_inputs().items()}

    return expected_likelihood + entropy, sample_stochastic_inputs

                                    
