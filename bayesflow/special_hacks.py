import numpy as np
import tensorflow as tf

"""

Contains hacks to compute special functions within TensorFlow graphs, e.g., by explicitly representing a power series. Should be phased out when TF actually implements native support for special functions. 
"""


def gammaln(z, n=14):
    # implement the lanczos approximation as described in numerical recipes 6.1

    assert (n <= 14)
    
    cof = (57.1562356658629235,-59.5979603554754912,
           14.1360979747417471,-0.491913816097620199,.339946499848118887e-4,
           .465236289270485756e-4,-.983744753048795646e-4,.158088703224912494e-3,
           -.210264441724104883e-3,.217439618115212643e-3,-.164318106536763890e-3,
           .844182239838527433e-4,-.261908384015814087e-4,.368991826595316234e-5)

    tmp = z+5.24218750000000000
    tmp = (z+0.5)*tf.log(tmp)-tmp
    ser = tf.Variable(0.999999999999997092)
    for j in range(n):
        ser = ser + cof[j]/(z+j+1.0)
        
    return tmp+tf.log(2.5066282746310005*ser/z)


    
