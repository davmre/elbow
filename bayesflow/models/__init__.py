import numpy as np
import tensorflow as tf

import uuid

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

    def __init__(self, shape=None, minibatch_scale_factor = None,
                 name=None, model="auto", **kwargs):

        if name is None:
            name = str(uuid.uuid4().hex)[:6]
            print "constructed name", name
        self.name = name
        
        self.minibatch_scale_factor = minibatch_scale_factor

        self.shape = shape
            
        # store map of input local names to the canonical names, that
        # is, (node, name) pairs -- modeling those params.
        self.inputs_random = {}
        self.inputs_nonrandom = {}
        for input_name, default_constructor in self.inputs().items():
            # inputs can be provided as constants, or nodes modeled by bayesflow distributions.
            if isinstance(kwargs[input_name], ConditionalDistribution):
                self.inputs_random[input_name] = kwargs[input_name]
            elif kwargs[input_name] is not None:
                # if inputs are provided as TF or numpy values, just store that directly
                tf_value = tf.convert_to_tensor(kwargs[input_name], dtype=tf.float32)
                self.inputs_nonrandom[input_name] = tf_value
            elif kwargs[input_name] is None:
                # free inputs will be optimized over
                self.inputs_nonrandom[input_name] = default_constructor(shape=self._input_shape(input_name))
                
        # if not already specified, compute the shape of the output at
        # this node as a function of its inputs
        if shape is None:
            input_shapes = {name + "_shape": node.shape for (name, node) in self.inputs_random.items()}
            input_shapes.update({name + "_shape": tnode.get_shape() for (name, tnode) in self.inputs_nonrandom.items()})
            self.shape = self._compute_shape(**input_shapes)

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

        # define a canonical 'sample' for this variable
        # in terms of canonical samples for the input variables.
        # this is useful when defining variational posteriors,
        # since passing a posterior automatically passes a
        # sample that we can use in monte carlo objectives. 
        all_random_sources = {}
        input_samples = {}
        for param, node in self.inputs_random.items():
            input_samples[param] = node._sampled
            all_random_sources.update(node._sampled_source)
        self._sampled, my_source = self._parameterized_sample(**input_samples)
        all_random_sources.update(my_source)
        self._sampled_source = all_random_sources
        
    def input_val(self, input_name):
        # return a (Monte Carlo estimate of) the value for the given input.
        # This allows distributions to easily return their parameters, for example.
        # Note these estimates depend on a random source function, which we do *not*
        # return here, but could easily be fulfilled from self._sampled at this node.
        if input_name in self.inputs_nonrandom:
            return self.inputs_nonrandom[input_name]
        else:
            return self.inputs_random[input_name]._sampled

    def _parameterized_logp(self, *args, **kwargs):
        """
        Compute the log probability using the values of all fixed input
        parameters associated with this graph node.
        """
        kwargs.update(self.inputs_nonrandom)
        return self._logp(*args, **kwargs)

    def _parameterized_sample(self, *args, **kwargs):
        kwargs.update(self.inputs_nonrandom)
        return self._sample(*args, **kwargs)

    def _entropy(self, *args, **kwargs):
        return self._parameterized_logp(*args, result=self._sample, **kwargs)
    
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
            samples[key] = qdist._sampled
        return self._parameterized_logp(**samples)

    def _optimized_params(self, sess, feed_dict=None):
        optimized_params = {}
        for name, node in self.inputs_nonrandom.items():
            optimized_params[name] = sess.run(node, feed_dict = None)
        return optimized_params

    def attach_q(self, q):
        assert(self.model is not None)
        self.model.marginalize(self, q)

    def observe(self, val):
        assert(self.model is not None)
        self.model.observe(self, val)

    def sample(self):
        assert(self.model is not None)
        return self._sampled
    
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.__class__.__name__) + "_" + self.name

class WrapperNode(ConditionalDistribution):
    """
    Lifts a Tensorflow graph node representing a point value to a ConditionalDistribution. 
    This implements the 'return' operation of the probability monad. 
    """

    def __init__(self, tf_value, **kwargs):
        self.tf_value = tf_value
        self.mean = tf_value
        self.variance = tf.zeros_like(self.mean, name="variance")
        
        shape = tf_value.get_shape()
        super(WrapperNode, self).__init__(shape=shape, tf_value=tf_value, **kwargs)
        
    def inputs(self):
        return {"tf_value": None}
        
    def _sample(self, tf_value):
        return tf_value, {}

    def _logp(self, **kwargs):
        return tf.constant(0.0, dtype=tf.float32), {}

    def _entropy(self, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)


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
