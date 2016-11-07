import numpy as np
import tensorflow as tf

def concrete_shape(shape):
    if isinstance(shape, tuple):
        return shape
    elif isinstance(shape, tf.TensorShape):
        return tuple([d.value for d in shape])
    else:
        raise Exception("don't know how to interpret %s as a tensor shape" % shape)
    
def extract_shape(t):

    if t.get_shape():
        pass
    shape = tuple([d.value for d in t.get_shape()])
    if len(shape)==1 and shape[0] is None:
        shape = ()
    return shape

def logsumexp(x1, x2):
    shift = tf.maximum(x1, x2)
    return tf.log(tf.exp(x1 - shift) + tf.exp(x2-shift)) + shift

def triangular_inv(L):
    eye = tf.diag(tf.ones_like(tf.diag_part(L)))
    invL = tf.matrix_triangular_solve(L, eye)
    return invL

def broadcast_shape(**shapes):
    result = None
    xs = [np.empty(shape) for shape in shapes.values()]
    return np.broadcast(*xs).shape
    

def differentiable_sq_singular_vals(A):
    # the standard tensorflow SVD repr isn't differentiable,
    # but if we fix the rotation we can get an approximate
    # derivative
    d, u, v = tf.svd(A)
    vv = tf.stop_gradient(v)
    ud = tf.matmul(A, vv)
    
    dd = tf.reduce_sum(tf.square(ud), 0)
    return dd
