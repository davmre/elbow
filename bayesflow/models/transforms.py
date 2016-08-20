import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import ConditionalDistribution
from bayesflow.models.q_distributions import QDistribution


"""
The current story for deterministic dependencies and transformations is really ugly.
Defining a transformed variable requires first defining the original variable,
then creating a new dependent variable, then attaching a Q distribution 
to that dependent variable to pass through the values from the parent Q distribution. 

This basically works but it's a hack and I need to find a more elegant solution. 

"""

class DeterministicTransform(ConditionalDistribution):

    def __init__(self, A, **kwargs):
        super(DeterministicTransform, self).__init__(A=A, **kwargs)
        
    def inputs(self):
        return {"A": None}
        
    def _logp(self, result, A):
        return tf.constant(0.0, dtype=tf.float32)

    def _entropy(self, A):
        return tf.constant(0.0, dtype=tf.float32)
        
    def _compute_dtype(self, A_dtype):
        return A_dtype

    def deterministic(self):
        return True

    def marginalize(self, qdist):
        raise Exception("cannot attach an explicit Q distribution to a deterministic transform. attach to the parent instead!")


class PointwiseTransformedMatrix(DeterministicTransform):

    def __init__(self, A, transform, implicit=False, **kwargs):

        # cases:
        # a) we want to construct an encapsulated ConditionalDistribution
        # that composes A with the transform. Here A exists outside
        # the graph and is just a convenience object used in the
        # construction. In that case we want the *current* object to
        # explicitly call out to A for sampling, entropy, etc

        # b) we want to add a deterministic variable to a graph that already contains
        #    the parent A. In this case the conditional distribution is really a delta,
        #    and the joint distribution p(A, B) is just p(A)delta(A==B). So the log 
        #    jacobian does not appear. This is the "implicit" case. 
        
        self.implicit = implicit
        self.transform=transform
        super(PointwiseTransformedMatrix, self).__init__(A=A, **kwargs)
        
    def _compute_shape(self, A_shape):
        return A_shape

    def _sample(self, A):
        tA, log_jacobian = self.transform(A)
        # HACK
        self._sampled_log_jacobian = log_jacobian
        return tA
    
    def _entropy(self, *args, **kwargs):
        if self.implicit:
            return tf.constant(0.0, dtype=tf.float32)
        else:
            return self.inputs_random["A"]._sampled_entropy + self._sampled_log_jacobian

    def _default_variational_model(self):
        # todo figure out a global view of how I should put Q distributions on deterministic variables
        pass
            
    
class TransposeQDistribution(QDistribution):
    def __init__(self, parent_q):
        N, D = parent_q.output_shape
        transposed_shape = (D, N)
        self.parent_q = parent_q
        
        super(TransposeQDistribution, self).__init__(shape=transposed_shape)

        for param in parent_q.params():
            self.__dict__[param] = tf.transpose(parent_q.__dict__[param])

        self.sample = tf.transpose(parent_q.sample)
            
        # HACKS
        try:
            self.variance = tf.transpose(parent_q.variance)
        except:
            pass
        try:
            self.stddev = tf.transpose(parent_q.stddev)
        except:
            pass
        try:
            self.mean = tf.transpose(parent_q.mean)
        except:
            pass
        
    def sample_stochastic_inputs(self):
        return self.parent_q.sample_stochastic_inputs()
        
    def entropy(self):        
        return tf.constant(0.0, dtype=tf.float32)

    def initialize_to_value(self, x):
        self.parent_q.initialize_to_value(x.T)
    
class Transpose(DeterministicTransform):
    def __init__(self, A, **kwargs):
        super(Transpose, self).__init__(A=A, **kwargs)
        
    def _sample(self, A):
        return A.T
    
    def _compute_shape(self, A_shape):
        N, D = A_shape
        return (D, N)

    def attach_q(self, qdist):
        """
        Try to attach a transformed Q to the parent instead...
        """

        parent = self.input_nodes["A"]
        parent_q = TransposeQDistribution(qdist)
        parent.attach_q(parent_q)

        self._q_distribution = qdist
        #super(Transpose, self).attach_q(qdist)
        
    def default_q(self):            
        parent_q = self.input_nodes["A"].q_distribution()
        return TransposeQDistribution(parent_q)
        
"""
class PointwiseTransformedQDistribution(QDistribution):
    def __init__(self, parent_q, transform, implicit=False):

        super(PointwiseTransformedQDistribution, self).__init__(shape=parent_q.output_shape)
        self.sample, self.log_jacobian = transform(parent_q.sample)

        self.implicit = implicit
        self.parent_q = parent_q
        
    def sample_stochastic_inputs(self):
        if self.implicit:
            return {}
        else:
            return self.parent_q.sample_stochastic_inputs()
        
    def entropy(self):
        if self.implicit:
            return tf.constant(0.0, dtype=tf.float32)
        else:
            return self.parent_q.entropy() + self.log_jacobian
"""
