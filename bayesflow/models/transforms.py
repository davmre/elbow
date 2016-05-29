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
        return ("A")
        
    def _logp(self, result, A):
        return tf.constant(0.0, dtype=tf.float32)
        
    def _compute_dtype(self, A_dtype):
        return A_dtype

    def attach_q(self, qdist):
        raise Exception("cannot attach an explicit Q distribution to a deterministic transform. attach to the parent instead!")

    
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
        

class PointwiseTransformedQDistribution(QDistribution):
    def __init__(self, parent_q, transform, implicit=False):

        super(PointwiseTransformedQDistribution, self).__init__(shape=parent_q.output_shape)
        self.sample, self.log_jacobian = transform(parent_q.sample)

        # an "implicit" transformation is a formal object associated with a deterministic
        # transformation in the model, where the parent q distribution is associated with
        # the untransformed variable. In this case the ELBO expectations are with respect
        # to the parent Q distribution, so the jacobian of the transformation does not appear.
        # By contrast, a non-implicit use of a transformed Q distribution would be to create
        # a new type of distribution (eg, lognormal by exponentiating a Gaussian parent)
        # that could then itself be associated with stochastic variables in the graph.
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
        

    
class PointwiseTransformedMatrix(DeterministicTransform):

    def __init__(self, A, transform, **kwargs):
        self.transform=transform
        super(PointwiseTransformedMatrix, self).__init__(A=A, **kwargs)        
        
    def _compute_shape(self, A_shape):
        return A_shape

    def _sample(self, A):
        tA, _ = self.transform(A)
        return tA
    
    def attach_q(self, qdist):
        raise Exception("cannot attach an explicit Q distribution to a deterministic transform. attach to the parent instead!")

    def default_q(self):
        parent_q = self.input_nodes["A"].q_distribution()
        return PointwiseTransformedQDistribution(parent_q, self.transform, implicit=True)
