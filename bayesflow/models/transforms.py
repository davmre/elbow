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
            


class Transform(object):

    @staticmethod
    def transform(return_log_jac=False, **kwargs):
        raise Exception("abstract base class")

    @staticmethod
    def inverse(transformed):
        raise Exception("not invertible!")

    @staticmethod
    def output_shape(input_shape):
        # default to assuming a pointwise transform
        return input_shape
    
class Logit(Transform):

    @staticmethod
    def transform(x, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_logit_input")
        transformed = 1.0 / (1 + tf.exp(-x))
        
        if return_log_jac:
            jacobian = transformed * (1-transformed)
            if clip_finite:
                jacobian = tf.clip_by_value(jacobian, 1e-45, 1e38, name="clipped_jacobian")
            log_jacobian = tf.reduce_sum(tf.log(jacobian))
            return transformed, log_jacobian
        else:
            return transformed

    @staticmethod
    def inverse(transformed):
        x = tf.log(1./transformed - 1.0)
        return x

class Normalize(Transform):

    @staticmethod
    def transform(x_positive, return_log_jac=False, **kwargs):
        n = util.extract_shape(x_positive)[0]
        Z = tf.reduce_sum(x_positive)
        transformed = x_positive / Z
        if return_log_jac:
            log_jacobian = -n * tf.log(Z)
            return transformed, log_jacobian
        else:
            return transformed

class Exp(Transform):

    @staticmethod
    def transform(x, return_log_jac=False, clip_finite=True):
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_exp_input")

        transformed = tf.exp(x)
        if return_log_jac:
            log_jacobian = tf.reduce_sum(x)
            return transformed, log_jacobian
        else:
            return transformed

class Square(Transform):

    @staticmethod
    def transform(x, return_log_jac=False, **kwargs):
        transformed = x * x

        if return_log_jac:
            log_jacobian = tf.reduce_sum(tf.log(x)) + np.log(2)
            return transformed, log_jacobian
        else:
            return transformed

class Sqrt(Transform):

    @staticmethod
    def transform(x, return_log_jac=False, **kwargs):
        transformed = tf.sqrt(x)
        if return_log_jac:
            jacobian = .5/transformed
            log_jacobian = tf.reduce_sum(tf.log(jacobian))
            return transformed, log_jacobian
        else:
            return transformed

        
class Reciprocal(Transform):

    @staticmethod
    def transform(x, return_log_jac=False, clip_finite=True):
        if clip_finite:
            # caution: assumes input is positive
            x = tf.clip_by_value(x, 1e-38, 1e38, name="clipped_reciprocal_input")
            
        nlogx = -tf.log(x)
        transformed = tf.exp(nlogx)
        
        if return_log_jac:
            log_jacobian = 2*tf.reduce_sum(nlogx)
            return transformed, log_jacobian
        else:
            return transformed

def chain_transforms(*transforms):
    class Chain(Transform):

        @staticmethod
        def transform(x, return_log_jac=False):
            log_jacs = []
            for transform in transforms:
                if return_log_jac:
                    x, lj = transform.transform(x, return_log_jac=return_log_jac)
                    log_jacs.append(lj)
                else:
                    x = transform.transform(x, return_log_jac=return_log_jac)
            if return_log_jac:
                return x, tf.reduce_sum(tf.pack(log_jacs))
            else:
                return x

        @staticmethod
        def inverse(transformed):
            for transform in transforms[::-1]:
                transformed = transform.inverse(transform)
            return transform

        @staticmethod
        def output_shape(input_shape):
            for transform in transforms:
                input_shape = transform.output_shape(input_shape)
            return input_shape
        
    return Chain

Reciprocal_Sqrt = chain_transforms(Reciprocal, Sqrt)
Reciprocal_Square = chain_transforms(Reciprocal, Square)
Exp_Reciprocal = chain_transforms(Exp, Reciprocal)
Simplex_Raw = chain_transforms(Exp, Normalize)

class Transpose(Transform):
    # todo should there be a special property for permutation
    # transforms, so that we also transform means, variances, etc?

    @staticmethod
    def transform(x, return_log_jac=False):
        transformed = tf.transpose(x)
        if return_log_jac:
            return transformed, 0.0
        else:
            return transformed

    @staticmethod
    def inverse(transformed):
        return tf.transpose(transformed)

    @staticmethod
    def output_shape(input_shape):
        N, M = input_shape
        return (M, N)


