import numpy as np
import tensorflow as tf

from util import concrete_shape
from transforms import Simplex, Logit
"""
Utility methods for defining constrained variables as transforms of an unconstrained parameterization. 
"""

def unconstrained(shape=None, init=None, name=None):

    if init is None:
        shape = concrete_shape(shape)
        init = np.float32(np.random.randn(*shape))
        
    val = tf.Variable(init, name=name)
    return val

def simplex_constrained(shape=None, init_log=None, name=None):

    if init_log is None:
        shape = concrete_shape(shape)
        init_log = np.float32(np.ones(shape) * -10)

    log_value = tf.Variable(init_log, name= "log_"+name if name is not None else None)
    return Simplex.transform(log_value)

def unit_interval(shape=None, init_log=None, name=None):
    # Defines a matrix each element of which is in the unit interval.
    # This is different from simplex_constrained which defines a
    # vector guaranteed to be in the unit simplex.

    if init_log is None:
        shape = concrete_shape(shape)
        init_log = np.float32(np.ones(shape) * -10)

    log_value = tf.Variable(init_log, name= "log_"+name if name is not None else None)
    return Logit.transform(log_value)

def positive_exp(shape=None, init_log=None, name=None):
    # a Tensor of values that are pointwise positive, represented by an exponential
    
    if init_log is None:
        shape = concrete_shape(shape)
        init_log = np.float32(np.ones(shape) * -10)
    
    log_value = tf.Variable(init_log, name= "log_"+name if name is not None else None)
    pos_value = tf.exp(tf.clip_by_value(log_value, -42, 42), name=name)
    return pos_value

def psd_matrix(shape=None, init=None, name=None):
    assert(init is None) # TODO figure out init semantics
    n, n2 = shape
    assert(n==n2)
    init = np.float32(np.eye(n) * 0.1)
    # TODO figure out lower triangular parameterization
    
    
    A = tf.Variable(init, name="latent_"+name if name is not None else None)
    psd = tf.matmul(tf.transpose(A), A, name=name)
    return psd

def psd_matrix_small(shape=None, init=None, name=None):
    assert(init is None) # TODO figure out init semantics
    n, n2 = shape
    assert(n==n2)
    init = np.float32(np.eye(n) * 1e-6)
    # TODO figure out lower triangular parameterization
    
    A = tf.Variable(init, name="latent_"+name if name is not None else None)
    psd = tf.matmul(tf.transpose(A), A, name=name)
    return psd

def psd_diagonal(shape=None, init=None, name=None):
    assert(init is None) # TODO figure out init semantics
    n, n2 = shape
    assert(n==n2)
    init = np.float32(np.zeros(n))
    
    latent_diag = tf.Variable(init, name="latent_"+name if name is not None else None)
    psd = tf.diag(tf.exp(latent_diag), name=name)
    return psd

def orthogonal_columns(shape=None, name=None, normalize=False, sort_columns=False, separate_norms=False):
    
    # Parameterizes a matrix by a Gram-Schmidt orthogonalization process.
    # That is, each column is defined by a set of 'latent' elements, from which
    # we subtract the projections of all previous columns.
    
    n, d = shape

    cols = []
    col_sqnorms = []
    for i_col in range(d):
        col_params = n - i_col
        init_col = np.float32(np.random.randn(col_params,) )
        latent_col = tf.Variable(init_col, name="orthog_col%d" % d)
        col = tf.pad(latent_col, [[0, i_col],])

        for prev_col, prev_sqnorm in zip(cols, col_sqnorms):
            proj = tf.reduce_sum(col * prev_col) / prev_sqnorm * prev_col
            col -= proj
        sqnorm = tf.reduce_sum(col*col)
        cols.append(col)
        col_sqnorms.append(sqnorm)

    orthog = tf.pack(cols, axis=1)
    sqnorms = tf.pack(col_sqnorms)

    if normalize:
        return orthog / tf.sqrt(sqnorms)
    elif separate_norms:
        norms = tf.sqrt(sqnorms)        
        normalized = orthog / norms

        colnorms = tf.exp(tf.Variable(np.float32(np.random.randn(d))))
        return normalized*colnorms
        
    elif sort_columns:
        norms = tf.sqrt(sqnorms)        
        normalized = orthog / norms
        
        logits = tf.Variable(np.float32(np.random.randn(d)))
        scalings = 1.0/(1+tf.exp(-logits))
        cum_scalings = tf.cumprod(scalings)

        total_norm = tf.exp(tf.Variable(np.float32(np.random.randn())))
        cum_scalings *= total_norm
        
        #newnorms = tf.reverse(cum_unif, [True,])
        return normalized * cum_scalings
    else:
        return orthog
            

