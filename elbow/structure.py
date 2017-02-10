import tensorflow as tf
import numpy as np

from elbow import ConditionalDistribution
from transforms import DeterministicTransform, UnaryTransform, Transform

class PackRVs(DeterministicTransform):

    def __init__(self, *rvs, **kwargs):
        self.n_rvs = len(rvs)
        self.rv_keys = ["inp%03d" % i for i in range(self.n_rvs)]
        inp_dict = {k: rv for (k, rv) in zip(self.rv_keys, rvs)}
        kwargs.update(inp_dict)
        super(PackRVs, self).__init__(**kwargs)

    def inputs(self):
        return {k: None for k in self.rv_keys}

    def _compute_shape(self, **inp_shapes):
        global_shape = None
        for k, shape in inp_shapes.items():
            # all variables should have the same shape outside of the first row
            if global_shape is None:
                global_shape = shape
            else:
                assert(shape == global_shape)

        return (len(inp_shapes),) + global_shape

    def _sample(self, **inputs):
        sorted_inputs = [v for (k, v) in sorted(inputs.items())]
        return tf.pack(sorted_inputs)

    def _inference_networks(self, q_result):
        networks = {}
        rvs = unpackRV(q_result)
        networks = {k: n for (k, n) in zip(sorted(self.inputs_random.keys()), rvs)}
            
        return networks

    def default_q(self):
        qs = [rv.q_distribution() for (k, rv) in sorted(self.inputs_random.items())]
        return PackRVs(*qs, name="q_" + self.name)
    
def unpackRV(rv, axis=0):
    """
    Convert an RV of shape (N, :, :, ...) into N rvs of shape (:, :, ...)
    (or analogously for other axis!=0). 
    """
    
    noutputs = rv.shape[axis]

    transformedRVs = []
    for output in range(noutputs):
        transform = unpack_transform(idx=output, axis=axis)
        transformedRVs.append(UnaryTransform(rv, transform))
        
    return transformedRVs


def unpack_transform(idx, axis=0):
    # given A, returns A[idx, :] if axis=0,
    # or analogously for other axis
    class Unpack(Transform):
        @classmethod
        def transform(cls, x, return_log_jac=False):
            transformed = tf.unpack(x, axis=axis)[idx]
            if return_log_jac:
                return transformed, 0.0
            else:
                return transformed

        @classmethod
        def output_shape(cls, input_shape):
            return input_shape[:axis] + input_shape[axis+1:]

        @classmethod
        def is_structural(cls):
            return True
        
    return Unpack

def split_at_row(rv, row_idx, name="split"):
    """
    Given an RV representing a matrix, return two RVs representing 
    the initial row_idxs rows and all following rows, respectively.
    """
    
    n, m = rv.shape
    t1 = slice_transform((0,0), (row_idx, m))
    t2 = slice_transform( (row_idx, 0), (n-row_idx, m))
    
    rv1 = UnaryTransform(rv, t1, name="split_%s_1" % rv.name)
    rv2 = UnaryTransform(rv, t2, name="split_%s_2" % rv.name)
    return rv1, rv2

def slice_transform(begin, size):
    # given A, returns A[idx, :] if axis=0,
    # or analogously for other axis
    class Slice(Transform):
        @classmethod
        
        def transform(cls, x, return_log_jac=False):
            transformed = tf.slice(x, begin, size)
            
            if return_log_jac:
                return transformed, 0.0
            else:
                return transformed

        @classmethod
        def output_shape(cls, input_shape):
            return size

        @classmethod
        def is_structural(cls):
            return True

    return Slice

def reshape_transform(new_shape, old_shape=None):
    class Reshape(Transform):
        @classmethod
        def transform(cls, x, return_log_jac=False):
            transformed = tf.reshape(x, new_shape)            
            if return_log_jac:
                return transformed, 0.0
            else:
                return transformed

        @classmethod
        def output_shape(cls, input_shape):
            a = np.empty(input_shape)
            return np.reshape(a, new_shape).shape

        @classmethod
        def input_shape(cls, output_shape):
            if old_shape is not None:
                a = np.empty(output_shape)
                return np.reshape(a, old_shape).shape
            else:
                raise Exception("must pass old_shape explicitly to allow inverse reshape transform")

        @classmethod
        def inverse(cls, x, return_log_jac=False):
            if old_shape is None:
                raise Exception("must pass old_shape explicitly to allow inverse reshape transform")

            transformed = tf.reshape(x, old_shape)            
            if return_log_jac:
                return transformed, 0.0
            else:
                return transformed
            
        @classmethod
        def is_structural(cls):
            return True

    return Reshape
    
Flatten = reshape_transform(new_shape=(-1,))
