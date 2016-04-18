import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util


from bayesflow.models import ConditionalDistribution
from bayesflow.models.q_distributions import PointwiseTransformedQDistribution

class PointwiseTransformedMatrix(ConditionalDistribution):

    def __init__(self, A, transform, **kwargs):
        self.transform=transform
        super(PointwiseTransformedMatrix, self).__init__(A=A, **kwargs)        
        
    def inputs(self):
        return ("A")
        
    def _sample(self, A):
        tA, _ = self.transform(A)
        return tA
    
    def _logp(self, result, A):
        return tf.constant(0.0, dtype=tf.float32)
    
    def _compute_shape(self, A_shape):
        return A_shape
        
    def _compute_dtype(self, A_dtype):
        return A_dtype

    def attach_q(self):
        raise Exception("cannot attach an explicit Q distribution to a deterministic transform. attach to the parent instead!")

    def __getattr__(self, name):
        # hack to generate the Q distribution when it's first requested. we can't do this at initialization
        # time since the parent might not have a Q distribution attached yet.
        
        if name=="q_distribution":
            parent_q = self.input_nodes["A"].q_distribution
            self.q_distribution = PointwiseTransformedQDistribution(parent_q, self.transform, implicit=True)
            return self.q_distribution

        raise AttributeError(name)
                
