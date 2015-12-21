import os
import numpy as np
import sys
import cPickle as pickle
import time

from util import batch_generator, get_mnist

import tensorflow as tf
import bayesflow as bf

class VariationalAutoEncoder(object):

    def __init__(self, d_z=2, d_hidden=256, d_x=784, batch_size=128): 

        self.d_z, self.d_x, self.d_hidden, self.batch_size = d_z, d_x, d_hidden, batch_size
        
        def init_weights(*shape):
            return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))

        def init_zero_vector(n_out):
            return tf.Variable(tf.zeros((n_out,), dtype=tf.float32))

        def layer(inp, w, b):
            return tf.matmul(inp, w) + b

        def vae_encoder(X, w3, w4, w5, b3, b4, b5):
            h1 = tf.nn.tanh(layer(X, w3, b3))
            mu = layer(h1, w4, b4)
            sigma2 = tf.exp(layer(h1, w5, b5))
            return mu, sigma2

        def vae_decoder_bernoulli(z, w1, w2, b1, b2):
            h1 = tf.nn.tanh(layer(z, w1, b1))
            h2 = tf.nn.sigmoid(layer(h1, w2, b2))
            return h2

        self.batch_X = tf.placeholder(tf.float32, shape=(batch_size, d_x))
        self.batch_eps = tf.placeholder(tf.float32, shape=(batch_size, d_z))

        w_encode_h3 = init_weights(d_x, d_hidden)
        w_encode_h4 = init_weights(d_hidden, 2)
        w_encode_h5 = init_weights(d_hidden, 2)
        b_encode_3 = init_zero_vector(d_hidden)
        b_encode_4 = init_zero_vector(d_z)
        b_encode_5 = init_zero_vector(d_z)

        w_decode_h = init_weights(d_z, d_hidden)
        w_decode_h2 = init_weights(d_hidden, d_x)
        b_decode_1 = init_zero_vector(d_hidden)
        b_decode_2 = init_zero_vector(d_x)

        encode_params = [w_encode_h3, w_encode_h4, w_encode_h5,  b_encode_3, b_encode_4, b_encode_5]

        decode_params = [w_decode_h, w_decode_h2, b_decode_1, b_decode_2 ]        

        self.params = encode_params + decode_params
        
        self.mu_z, self.sigma2_z = vae_encoder(self.batch_X, *encode_params)
        self.z = self.batch_eps * tf.sqrt(self.sigma2_z) + self.mu_z
        self.y = vae_decoder_bernoulli(self.z, *decode_params)

        self.kl = tf.reduce_sum(bf.dists.gaussian_kl(self.mu_z, self.sigma2_z), 1)

        self.lps = tf.reduce_sum(bf.dists.bernoulli_log_density(self.batch_X, self.y), 1)
        self.elbo = tf.reduce_mean(self.lps - self.kl) #/ float(batch_size)

def train(seed=3):
    batch_size=128
    d_z = 2

    np.random.seed(seed)
    
    Xdata, ydata = get_mnist()
    Xtrain = Xdata[0:60000]
    sortinds = np.random.permutation(60000)
    Xtrain = Xtrain[sortinds]
    bigeps = np.random.randn(len(Xtrain), d_z)
    Xtest = Xdata[60000:70000]

    vae = VariationalAutoEncoder(batch_size=None)

    train_step = tf.train.AdamOptimizer(0.005).minimize(-vae.elbo)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i_epoch in xrange(100):
        tstart = time.time()
        bigeps = np.random.randn(len(Xtrain), d_z)
        for start in xrange(0, Xtrain.shape[0], batch_size):
            end = start+batch_size
            Xt = Xtrain[start:end]
            eeps = bigeps[start:end]

            feed_dict = {vae.batch_X: Xt,
                         vae.batch_eps: eeps}

            sess.run(train_step, feed_dict = feed_dict)
            
        elapsed = time.time() - tstart
        feed_dict[vae.batch_X] = Xtrain[:len(Xtest)]
        feed_dict[vae.batch_eps] = bigeps[:len(Xtest)]
        (mll,) = sess.run((vae.elbo,), feed_dict=feed_dict)
        print i_epoch, mll, elapsed
        
        w = sess.run(vae.params, feed_dict=feed_dict)
        with open("weights_%d.pkl" % i_epoch, 'wb') as f:
            pickle.dump(w, f)

def visualize():
    vae = VariationalAutoEncoder(batch_size=1)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    images = []
    image_zs = []
    for i_epoch in range(100):

        weights_file = "/Users/dmoore/Dropbox/projects/vae/weights_%d.pkl" % i_epoch
        with open(weights_file, 'rb') as f:
             w = pickle.load(f)

        w3, w4, w5, b3, b4, b5, w1, w2, b1, b2 = w
        feed_dict =  {}
        for (p, val) in zip(vae.params, w):
            feed_dict[p] = val

        zs = np.linspace(-3, 3, 10)
        for i in range(10):
            for j in range(10):
                zz = np.array((zs[i], zs[j])).reshape(1, 2)

                feed_dict[vae.z] = zz

                (decoded,) = sess.run((vae.y,), feed_dict = feed_dict)

                images.append(decoded.reshape(28, 28))
                image_zs.append(zz)
        np.save("images_%d.npy" % i_epoch, np.asarray(images))
        np.save("image_zs.npy", np.asarray(image_zs))
        print i_epoch
        
def animate():
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.animation as manimation

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    f = plt.figure(figsize=(10, 10))

    with writer.saving(f, "stuff.mp4", 100):
        for i_epoch in range(100):
            images =  np.load("images_%d.npy" % i_epoch)

            gs = gridspec.GridSpec(10, 10)

            k=0
            for i in range(10):
                for j in range(10):
                    ax = plt.subplot(gs[i, j])
                    img = images[k, :, :]
                    ax.imshow(img, cmap="gray")
                    ax.set_axis_off()
                    k += 1
            writer.grab_frame()
            print "frame", i_epoch
                        
if __name__ == "__main__":
    train()
    #visualize()
    #animate()





