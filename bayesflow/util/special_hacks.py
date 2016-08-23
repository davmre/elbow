import numpy as np
import tensorflow as tf
import scipy.special

"""

Contains hacks to compute special functions within TensorFlow graphs,
e.g., by explicitly representing a power series. To be phased out
when TF actually implements native support for special functions.
"""

def gammaln(x):
    # fast approximate gammaln from paul mineiro
    # http://www.machinedlearnings.com/2011/06/faster-lda.html
    # (also in fastapprox: https://code.google.com/p/fastapprox/)
    logterm = tf.log (x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tf.log (xp3)

def betaln(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x+y)

def _test():

    z = tf.placeholder(dtype=tf.float32, name="z")
    gammaln_z = gammaln(z)

    x = tf.placeholder(dtype=tf.float32, name="x")
    y = tf.placeholder(dtype=tf.float32, name="y")
    betaln_z = betaln(x, y)

    init = tf.initialize_all_variables()
        
    sess = tf.Session()
    sess.run(init)

    for i in range(10):
        # sample random input from lognormal distribution
        z_val = np.exp(np.random.randn() * 4)

        gamma_tf, = sess.run([gammaln_z,], feed_dict={z: z_val})
        gamma_scipy = scipy.special.gammaln(z_val)
        print z_val, gamma_tf, gamma_scipy

    for i in range(10):
        # sample random input from lognormal distribution
        x_val = np.exp(np.random.randn() * 4)
        y_val = np.exp(np.random.randn() * 4)

        beta_tf, = sess.run([betaln_z,], feed_dict={x: x_val, y: y_val})
        beta_scipy = scipy.special.betaln(x_val, y_val)
        print x_val, y_val, beta_tf, beta_scipy

if __name__ == "__main__":
    _test()
