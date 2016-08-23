import numpy as np
import tensorflow as tf

import bayesflow as bf
import util

from bayesflow.conditional_dist import ConditionalDistribution

class DeterministicTransform(ConditionalDistribution):
    """
    Define a random variable as the deterministic transform of another RV 
    already present in the model. Variables of this type are treated as a 
    special case by the joint model: they cannot be given their own Q
    distributions, but are automatically given a Q distribution corresponding 
    to a deterministic transform of the parent's Q distribution. 
    """
    
    def __init__(self, A, transform, **kwargs):
        self.transform=transform
        assert(isinstance(A, ConditionalDistribution))
        super(DeterministicTransform, self).__init__(A=A, **kwargs)

    def inputs(self):
        return {"A": None}

    def _sample(self, A):
        tA = self.transform.transform(A)
        return tA

    def _compute_shape(self, A_shape):
        return self.transform.output_shape(A_shape)

    def _compute_dtype(self, A_dtype):
        return A_dtype
    
    def _logp(self, result, A):
        return tf.constant(0.0, dtype=tf.float32)

    def _entropy(self, A):
        return tf.constant(0.0, dtype=tf.float32)
        
    def marginalize(self, qdist):
        raise Exception("cannot attach an explicit Q distribution to a deterministic transform. attach to the parent instead!")

    def default_q(self):
        q_A = self.inputs_random["A"].q_distribution()    
        return DeterministicTransform(q_A, self.transform)

class TransformedDistribution(ConditionalDistribution):

    """
    Define a new distribution as a deterministic transform of a given 
    source distribution. Unlike the DeterministicTransform, which transforms
    a RV already present in the model, this creates a new random variable
    having the density of the transformed distribution. 

    """
    
    def __init__(self, dist, transform, **kwargs):

        # we assume dist is an *instance* of a ConditionalDist class that is
        # not currently part of any model, but will have some set of
        # (random and/or nonrandom) inputs that will be absorbed into the
        # TransformedDistribution.
        # As a convenience, we allow passing in an abstract class, e.g. Gaussian,
        # and instantiate the class ourselves with arguments passed into
        # the TransformedDistribution
        if isinstance(dist, type):
            self.dist = dist(**kwargs)
        else:
            self.dist = dist

        self.transform=transform
        super(TransformedDistribution, self).__init__(**kwargs)

        del self._sampled
        self._sampled, self._sampled_log_jacobian = self.transform.transform(self.dist._sampled, return_log_jac=True)

    def _setup_inputs(self, **kwargs):
        self.inputs_random = self.dist.inputs_random
        self.inputs_nonrandom = self.dist.inputs_nonrandom

    def _setup_canonical_sample(self):
        self._sampled, self._sampled_log_jacobian = self.transform.transform(self.dist._sampled, return_log_jac=True)
        self._sampled_entropy = self.dist._sampled_entropy + self._sampled_log_jacobian
        
    def inputs(self):
        return self.dist.inputs()

    def _compute_shape(self, **kwargs):
        return self.transform.output_shape(self.dist.shape)

    def _compute_dtype(self, **kwargs):
        return self.dist.dtype
    
    def _sample(self, **kwargs):
        ds = self.dist._sample(**kwargs)
        return self.transform.transform(ds)

    def _logp(self, result, **kwargs):
        inverted, inverse_logjac = self.transform.inverse(result, return_log_jac=True)
        return self.dist._logp(inverted, **kwargs) - inverse_logjac
        
    def _entropy(self, *args, **kwargs):
        return self.dist._entropy(*args, **kwargs) + self._sampled_log_jacobian

    def _default_variational_model(self, **kwargs):
        dvm = self.dist._default_variational_model()
        return TransformedDistribution(dvm, self.transform)



#############################################################################
    
class Transform(object):

    """
    Abstract class representing a deterministic transformation of a matrix. 
    Subclasses should implement the transform() method, including the option
    to return the log jacobian determinant, and the output_shape() method. 
    Invertible transforms should also implement inverse() and input_shape(). 
    
    Transform objects contain no state and are never instantiated, all methods
    are static only. (we use Python classmethods to allow a static method to 
    call other static methods of the same class (used, e.g., in implementing 
    SelfInverseTransform). 
    """
    
    @classmethod
    def transform(cls, x, return_log_jac=False, **kwargs):
        raise NotImplementedError

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, **kwargs):
        raise NotImplementedError

    @classmethod
    def output_shape(cls, input_shape):
        # default to assuming a pointwise transform
        return input_shape

    @classmethod
    def input_shape(cls, output_shape):
        # default to assuming a pointwise transform
        return output_shape


class SelfInverseTransform(Transform):

    """
    Base class for transforms that are their own inverse (transpose, reciprocal, etc.).
    """
    
    @classmethod
    def inverse(cls, *args, **kwargs):
        return cls.transform(*args, **kwargs)

    @classmethod
    def input_shape(cls, output_shape):
        return cls.output_shape(output_shape)
    
class Logit(Transform):

    """
    Map from the real line to the unit interval using the logistic sigmoid fn. 
    """
    
    @classmethod
    def transform(cls, x, return_log_jac=False, clip_finite=True, **kwargs):
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

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, **kwargs):
        x = tf.log(1./transformed - 1.0)

        if return_log_jac:
            jacobian = transformed * (1-transformed)
            if clip_finite:
                jacobian = tf.clip_by_value(jacobian, 1e-45, 1e38, name="clipped_jacobian")
            log_jacobian = -tf.reduce_sum(tf.log(jacobian))
            return x, log_jacobian
        else:
            return x

class Normalize(Transform):

    """
    TODO this was written for vectors only, what are its semantics applied to matrices?
    """
    
    @classmethod
    def transform(cls, x_positive, return_log_jac=False, **kwargs):
        n = util.extract_shape(x_positive)[0]
        Z = tf.reduce_sum(x_positive)
        transformed = x_positive / Z
        if return_log_jac:
            log_jacobian = -n * tf.log(Z)
            return transformed, log_jacobian
        else:
            return transformed

class Exp(Transform):

    @classmethod
    def transform(cls, x, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_exp_input")

        transformed = tf.exp(x)
        if return_log_jac:
            log_jacobian = tf.reduce_sum(x)
            return transformed, log_jacobian
        else:
            return transformed

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            transformed = tf.clip_by_value(transformed, 1e-45, 1e38, name="clipped_log_input")
        x = tf.log(transformed)

        if return_log_jac:
            log_jacobian = -tf.reduce_sum(x)
            return x, log_jacobian
        else:
            return x
        
class Square(Transform):

    @classmethod
    def transform(cls, x, return_log_jac=False, **kwargs):
        transformed = tf.square(x)

        if return_log_jac:
            log_jacobian = tf.reduce_sum(tf.log(x)) + np.log(2)
            return transformed, log_jacobian
        else:
            return transformed

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, **kwargs):
        x = tf.sqrt(transformed)
        if return_log_jac:
            jacobian = .5/x
            log_jacobian = tf.reduce_sum(tf.log(jacobian))
            return x, log_jacobian
        else:
            return x

        
class Reciprocal(SelfInverseTransform):

    @classmethod
    def transform(cls, x, return_log_jac=False, clip_finite=True, **kwargs):
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


class Transpose(SelfInverseTransform):
    # todo should there be a special property for permutation
    # transforms, so that we also transform means, variances, etc?

    @classmethod
    def transform(cls, x, return_log_jac=False):
        transformed = tf.transpose(x)
        if return_log_jac:
            return transformed, 0.0
        else:
            return transformed

    @classmethod
    def output_shape(cls, input_shape):
        N, M = input_shape
        return (M, N)


def invert_transform(source):
    """
    Given a Transform class, return a class 
    with methods swapped to perform the inverse transform.
    """
    
    class Inverted(Transform):

        @classmethod
        def transform(cls, *args, **kwargs):
            return source.inverse(*args, **kwargs)

        @classmethod
        def inverse(cls, *args, **kwargs):
            return source.transform(*args, **kwargs)

        @classmethod
        def output_shape(cls, *args, **kwargs):
            return source.input_shape(*args, **kwargs)

        @classmethod
        def input_shape(cls, *args, **kwargs):
            return source.output_shape(*args, **kwargs)

    return Inverted

def chain_transforms(*transforms):
    
    class Chain(Transform):

        @classmethod
        def transform(cls, x, return_log_jac=False):
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

        @classmethod
        def inverse(cls, transformed):
            for transform in transforms[::-1]:
                transformed = transform.inverse(transform)
            return transform

        @classmethod
        def output_shape(cls, input_shape):
            for transform in transforms:
                input_shape = transform.output_shape(input_shape)
            return input_shape

        @classmethod
        def input_shape(cls, output_shape):
            for transform in transforms[::-1]:
                output_shape = transform.input_shape(output_shape)
            return output_shape

    return Chain

# define some common transforms by composing the base transforms defined above
Sqrt = invert_transform(Square)
Log = invert_transform(Exp)
Reciprocal_Sqrt = chain_transforms(Reciprocal, Sqrt)
Reciprocal_Square = chain_transforms(Reciprocal, Square)
Exp_Reciprocal = chain_transforms(Exp, Reciprocal)

Simplex_Raw = chain_transforms(Exp, Normalize)
class Simplex(Transform):
    # result is invariant to shifting the (logspace) input,
    # so we choose a shift to avoid overflow

    @classmethod
    def transform(cls, x, **kwargs):
        xmax = tf.reduce_max(x)
        return Simplex_Raw.transform(x-xmax, **kwargs)
