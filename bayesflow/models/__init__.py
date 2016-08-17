import numpy as np
import tensorflow as tf

import uuid

import bayesflow as bf
import bayesflow.util as util
from bayesflow.models.q_distributions import ObservedQDistribution, GaussianQDistribution

def sugar_fullname(name):
    """
    Allow omitting the output name when passing a node with a single output
    """
    
    if isinstance(name, ConditionalDistribution):
        outputs = name.outputs()
        if len(outputs) > 1:
            raise Exception("ConditionalDistribution %s has multiple outputs %s, please specify which one you want! Use a tuple (node, output_name)." % (name, outputs))
        elif len(outputs) == 0:
            raise Exception("ConditionalDistribution %s has no outputs!" % (name,))
        else:
            return (name, outputs[0])
    else:
        return name

def namestr(name):
    if isinstance(name, str):
        return name
    else:
        node, subname = name
        return str(node) + "_" + namestr(subname)
    
class ConditionalDistribution(object):
    """
    
    Generic object representing a conditional distribution: a single output conditioned on some set of inputs. 
    Unconditional distributions are the special case where the inputs are the empty set. 

    The inputs are assumed to be random variables described by their own respective distributions. 
    Thus the object graph implicitly represents a directed graphical model (Bayesian network).

    """

    def __init__(self, shape=None, minibatch_scale_factor = None,
                 name=None, model="auto", **kwargs):

        if name is None:
            name = str(uuid.uuid4().hex)[:6]
            print "constructed name", name
        self.name = name
        
        self.minibatch_scale_factor = minibatch_scale_factor

        if shape is not None:
            self.shape = shape
            
        # store map of input local names to the canonical names, that
        # is, (node, name) pairs -- modeling those params.
        self.inputs_random = {}
        self.inputs_nonrandom = {}
        for input_name, default_constructor in self.inputs().items():
            # inputs can be provided as constants, or nodes modeled by bayesflow distributions.
            if isinstance(kwargs[input_name], ConditionalDistribution):
                self.inputs_random[input_name] = sugar_fullname(kwargs[input_name])
            elif kwargs[input_name] is not None:
                # if inputs are provided as TF or numpy values, just store that directly
                tf_value = tf.convert_to_tensor(kwargs[input_name], dtype=tf.float32)
                self.inputs_nonrandom[input_name] = tf_value
            elif kwargs[input_name] is None:
                # free inputs will be optimized over
                self.inputs_nonrandom[input_name] = default_constructor(shape=self._input_shape(input_name))
                
        # if not already specified, compute the shape of the output at
        # this node as a function of its inputs
        if self.shape is None:
            input_shapes = {name + "_shape": node.shape for (name,node) in self.inputs_random.items()}
            input_shape.update({name + "_shape": tnode.get_shape() for (name, tnode) in self.inputs_nonrandom.items()})
            self.shape = self._compute_shape(**input_shapes)
        
        # compute the list of all ancestor nodes in the graph, by
        # merging the ancestor lists of the parent nodes.  Storing
        # this for every node is slightly inefficient, but makes for
        # simpler code by avoiding the need for a graph traversal when
        # constructing joint quantities like the ELBO. This could be
        # optimized if the ancestor lists ever become a performance
        # bottleneck.
        self.ancestors = set( [self,] + [ancestor for inp in self.inputs_random.values() if inp is not None for ancestor in inp[0].ancestors ] )
    
        self._sampled_value = None
        self._sampled_value_seed = None

        self._q_distribution = None

        # TODO do something sane here...
        self.dtype = tf.float32

        self.model = None
        if model == "auto":
            model = current_scope()
        if model is not None:
            # will set self.model on this rv
            model.extend(self)
        
    def outputs(self):
        return (self.name,)

    def _parameterized_logp(self, *args, **kwargs):
        """
        Compute the log probability using the values of all fixed input
        parameters associated with this graph node.
        """
        kwargs.update(self.inputs_nonrandom)
        return self._logp(*args, **kwargs)

    def _parameterized_sample(self, *args, **kwargs):
        """
        Compute the log probability using the values of all fixed input
        parameters associated with this graph node.
        """
        kwargs.update(self.inputs_nonrandom)
        return self._sample(*args, **kwargs)
    
    def _entropy_lower_bound(self, *args, **kwargs):
        """
        If a distribution defines an exact entropy, use that.
        """
        return self._entropy(*args, **kwargs)

    def _parameterized_entropy_lower_bound(self, *args, **kwargs):
        kwargs.update(self.inputs_nonrandom)
        return self._entropy_lower_bound(*args, **kwargs)
    
    def _expected_logp(self, **kwargs):
        # default implementation: compute E_q[ log p(x) ] as a Monte Carlo sample.
        samples = {}
        for (q_key, qdist) in kwargs.items():
            assert(q_key.startswith("q_"))
            key = q_key[2:]
            samples[key] = qdist.sample
        return self._logp(**samples)

    def _optimized_params(self, sess, feed_dict=None):
        optimized_params = {}
        for name, node in self.inputs_nonrandom.items():
            optimized_params[name] = sess.run(node, feed_dict = None)
        return optimized_params
            
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.__class__.__name__) + "_" + namestr(self.name) 



class WrapperNode(ConditionalDistribution):
    """
    Lifts a Tensorflow graph node representing a point value to a ConditionalDistribution. 
    This implements the 'return' operation of the probability monad. 
    """

    def __init__(self, tf_value, **kwargs):
        self.tf_value = tf_value
        shape = self.shape = tf_value.get_shape()
        super(WrapperNode, self).__init__(shape=shape, **kwargs)
        
        
        #self.ancestors = set()
        #self.inputs_random = {}
        #self.inputs_nonrandom = {}

        #print "WN name", name
        #if isinstance(name, tuple):
        #    import pdb; pdb.set_trace()

    def inputs(self):
        return {}
        
    def _sample(self):
        return {self.name: self.tf_value}, lambda : {}

    def _logp(self, **kwargs):
        return tf.constant(0.0, dtype=tf.float32), {}

    def _entropy(self, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)

    def outputs(self):
        return (self.name,)

_bf_jointmodel_stack_ = ()
class JMContext(object):
    def __init__(self, jm=None):
        if jm is None:
            from bayesflow.models.joint_model import JointModel
            jm = JointModel()
        self.jm = jm

    def __enter__(self):
        global _bf_jointmodel_stack_
        _bf_jointmodel_stack_ = _bf_jointmodel_stack_ + (self.jm,)        
        return self.jm
    
    def __exit__(self, type, value, traceback):
        global _bf_jointmodel_stack_
        _bf_jointmodel_stack_ = _bf_jointmodel_stack_[:-1]
        
def current_scope():
    global _bf_jointmodel_stack_
    return _bf_jointmodel_stack_[-1]
