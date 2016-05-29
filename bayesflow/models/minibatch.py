import numpy as np
import tensorflow as tf

import bayesflow as bf
import bayesflow.util as util

from bayesflow.models import ConditionalDistribution
from bayesflow.models.q_distributions import QDistribution

class PackModels(ConditionalDistribution):

    """
    Given a list of ConditionalDistributions each modeling identically-shaped RVs,
    return a ConditionalDistribution modeling the array that packs all their values
    together (following tf.pack). 

    TODO: some sort of generic framework for reshaping / other trivial transforms
    that generalizes this (and Transpose, etc) and doesn't require writing a full 
    ConditionDistribution and QDistribution with tons of redundant code for
    each transform. 
    """
    
    def __init__(self, model_list, **kwargs):

        self.model_list = model_list
        self.n = len(model_list)

        model_ids = self.inputs()
        d = {mid: model for (mid, model) in zip(model_ids, model_list)}
        kwargs.update(d)
        
        super(PackModels, self).__init__(**kwargs)

    def inputs(self):
        return ["packed_%05d" % i for i in range(self.n)]

    def _compute_shape(self, **kwargs):
        base_shape = None
        for modelid, model_shape in kwargs.items():
            if base_shape is None:
                base_shape = tuple(model_shape)
            else:
                assert(base_shape == model_shape)
        return (self.n,) + base_shape
                
    def _compute_dtype(self, **kwargs):
        base_dtype = None
        for modelid, model_dtype in kwargs.items():
            if base_dtype is None:
                base_dtype = model_dtype
            else:
                assert(base_dtype == model_dtype)
        return base_dtype

    def _logp(self, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)
    
    def _sample(self, **kwargs):
        vs = [v for (k,v) in sorted(kwargs.items())]
        return np.asarray(vs)

    def attach_q(self):
        raise Exception("cannot attach an explicit Q to a packed model, attach piecewise to the parents instead!")
        
    def default_q(self):            
        parent_qs = [m.q_distribution() for m in self.model_list]
        return PackedQDistribution(parent_qs)
        
class PackedQDistribution(QDistribution):

    def __init__(self, parent_qs):

        # TODO: this is redundant with the checks in PackModels
        self.n = len(parent_qs)
        base_shape = parent_qs[0].output_shape
        for pq in parent_qs:
            assert (pq.output_shape == base_shape)
        
        packed_shape = (self.n,) + base_shape

        self.parent_qs = parent_qs
        
        super(PackedQDistribution, self).__init__(shape=packed_shape)

        self.sample = tf.pack([pq.sample for pq in parent_qs])
            
        # HACKS
        try:
            self.variance = tf.pack([pq.variance for pq in parent_qs])
        except:
            pass
        try:
            self.stddev = tf.pack([pq.stddev for pq in parent_qs])
        except:
            pass
        try:
            self.mean = tf.pack([pq.mean for pq in parent_qs])
        except:
            pass

    def sample_stochastic_inputs(self):
        # parent qs will sample their own inputs
        return {}

    def entropy(self):        
        return tf.constant(0.0, dtype=tf.float32)

