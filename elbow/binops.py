import numpy as np
import tensorflow as tf


import util

from conditional_dist import ConditionalDistribution
from transforms import DeterministicTransform


class BinaryTransform(DeterministicTransform):
    """
    Define a random variable as the deterministic transform of another RV 
    already present in the model. Variables of this type are treated as a 
    special case by the joint model: they cannot be given their own Q
    distributions, but are automatically given a Q distribution corresponding 
    to a deterministic transform of the parent's Q distribution. 
    """
    
    def __init__(self, A, B, binop, **kwargs):
        self.binop = binop
        assert(isinstance(A, ConditionalDistribution))
        assert(isinstance(B, ConditionalDistribution))
        super(BinaryTransform, self).__init__(A=A, B=B, **kwargs)
            
    def inputs(self):
        d = {"A": None, "B": None}
        return d
    
    def _sample(self, A, B):
        tA = self.binop.combine(A, B)
        return tA

    def _compute_shape(self, A_shape, B_shape):
        return self.binop.output_shape(A_shape, B_shape)

    def _compute_dtype(self, A_dtype, B_dtype):
        return A_dtype
    
    def observe(self, observed_val):
        raise Exception("cannot observe value of a deterministic binary operator")
        
    def default_q(self):
        q_A = self.inputs_random["A"].q_distribution()
        q_B = self.inputs_random["B"].q_distribution()    
        return BinaryTransform(q_A, q_B, self.binop, name="q_"+self.name)

class CombinedDistribution(ConditionalDistribution):

    """
    Analogous to TransformedDistribution, but for binops. 

    Define a new distribution as the binary transform of two source
    distributions, and wraps/hides the input distribution, so that the
    graph only sees the combined distribution.

    """

    def __init__(self, A, B, binop, **kwargs):

        # we assume dist is an *instance* of a ConditionalDist class that is
        # not currently part of any model, but will have some set of
        # (random and/or nonrandom) inputs that will be absorbed into the
        # TransformedDistribution.
        # As a convenience, we allow passing in an abstract class, e.g. Gaussian,
        # and instantiate the class ourselves with arguments passed into
        # the TransformedDistribution
        if isinstance(A, type) or isinstance(B, type):
            raise NotImplementedError
        else:
            self.A = A
            self.B = B
            
        self.binop = binop
        super(CombinedDistribution, self).__init__(**kwargs)
        
    def _setup_inputs(self, **kwargs):

        inputs_random = {}
        inputs_nonrandom = {}
        for (k, v) in self.A.inputs_random.items():
            inputs_random["A_" + k] = v
        for (k, v) in self.B.inputs_random.items():
            inputs_random["B_" + k] = v

        for (k, v) in self.A.inputs_nonrandom.items():
            inputs_nonrandom["A_" + k] = v
        for (k, v) in self.B.inputs_nonrandom.items():
            inputs_nonrandom["B_" + k] = v

        self.inputs_random = inputs_random
        self.inputs_nonrandom = inputs_nonrandom
        return {}

    def _sample_and_entropy(self, **kwargs):
        a = self.A._sampled
        b = self.B._sampled
        assert(self.binop.is_structural())
        sample = self.binop.combine(a, b)
        entropy = self.A._sampled_entropy + self.B._sampled_entropy 
        return sample, entropy
    
    def inputs(self):
        inputs = {}
        for (k, v) in self.A.inputs().items():
            inputs["A_"+k] = v
        for (k, v) in self.B.inputs().items():
            inputs["B_"+k] = v
        return inputs

    def _deconstruct_args(self, kwargs):
        kwargs_A = {}
        kwargs_B = {}
        for (k, v) in kwargs.items():
            if k.startswith("A_"):
                kwargs_A[k[2:]] = v
            elif k.startswith("B_"):
                kwargs_B[k[2:]] = v
            else:
                raise Exception("unrecognized argument %s" % k)
        return kwargs_A, kwargs_B
                
    def _compute_shape(self, **kwargs):
        return self.binop.output_shape(self.A.shape, self.B.shape)

    def _sample(self, **kwargs):
        a = self.A._sampled
        b = self.B._sampled
        assert(self.binop.is_structural())
        sample = self.binop.combine(a, b)
        return sample
    
    def _logp(self, result, **kwargs):
        assert(self.binop.is_structural())
        a, b = self.binop.invert(result,
                                 a_shape=self.A.shape,
                                 b_shape=self.B.shape)  
        kwargs_A, kwargs_B = self._deconstruct_args(kwargs)
        return self.A._logp(a, **kwargs_A) + self.B._logp(b, **kwargs_B)
        
    #def _entropy(self, *args, **kwargs):
    #    raise NotImplementedError

    def default_q(self, **kwargs):
        dvmA = self.A.default_q()
        dvmB = self.B.default_q()
        return CombinedDistribution(dvmA, dvmB, self.binop, name="q_"+self.name)

    
    
#################################################
    
class BinOp(object):

    @classmethod
    def combine(cls, a, b, **kwargs):
        raise NotImplementedError

    @classmethod
    def output_shape(cls, a_shape, b_shape, **kwargs):
        raise NotImplementedError

    @classmethod
    def invert(cls, x, a_shape, b_shape, **kwargs):
        raise NotImplementedError

    @classmethod
    def is_structural(cls, **kwargs):
        return False

def concat_binop(concat_dim):
    class Concat(BinOp):

        @classmethod
        def combine(cls, a, b, **kwargs):
            return tf.concat(concat_dim, (a, b))

        @classmethod
        def output_shape(cls, a_shape, b_shape, **kwargs):
            for i, asi in enumerate(a_shape):
                if i != concat_dim:
                    assert(asi == b_shape[i])
            out_shape = np.copy(a_shape)
            out_shape[concat_dim] = a_shape[concat_dim] + b_shape[concat_dim]
            return out_shape

        @classmethod
        def invert(cls, x, a_shape, b_shape, **kwargs):
            begin = np.zeros(len(a_shape), dtype=np.int32)
            mid = np.zeros(len(a_shape), dtype=np.int32)
            mid[concat_dim] = a_shape[concat_dim]
            a = tf.slice(x, begin, a_shape)
            b = tf.slice(x, mid, b_shape)
            return a, b

        @classmethod
        def is_structural(cls, **kwargs):
            return True

        
    return Concat

VStack = concat_binop(concat_dim = 0)
HStack = concat_binop(concat_dim = 1)


def elementwise_binop(op):
    
    class ElementWise(BinOp):

        @classmethod
        def combine(cls, a, b, **kwargs):
            return op(a, b)

        @classmethod
        def output_shape(cls, a_shape, b_shape, **kwargs):
            assert((a_shape == b_shape).all())
            return a_shape

    return ElementWise

ElementWiseSum = elementwise_binop(lambda a, b: a + b)
ElementWiseProduct = elementwise_binop(lambda a, b: a * b)
ElementWiseSub = elementwise_binop(lambda a, b: a - b)
ElementWiseDiv = elementwise_binop(lambda a, b: a / b)
