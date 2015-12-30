import numpy as np
import tensorflow as tf

import dists

class MeanFieldInference(object):

    def __init__(self, joint_density, **jd_args):

        self.joint_density = joint_density
        self.jd_args = jd_args

        self.latents = {}
        
    def add_latent(self, name, init_mean=None, init_stddev=1e-6, transform=None):
        if init_mean is None:
            init_mean = np.random.randn()
        latent = {}
        latent["q_mean"] = tf.Variable(init_mean, name="%s_q_mean" % name)
        latent["q_stddev"] = tf.Variable(init_stddev, name="%s_q_stddev" % name)
        latent["q_entropy"] = dists.gaussian_entropy(stddev=latent["q_stddev"])
        latent["transform"] = transform
        self.latents[name] = latent

    def build_stochastic_elbo(self, n_eps=1):

        self.total_entropy = tf.add_n([d["q_entropy"] for d in self.latents.values()])
        
        stochastic_elbo_terms = []
        self.gaussian_inputs = []
        for i in range(n_eps):
            symbols = {}
            for name, latent in self.latents.items():
                eps = tf.placeholder(dtype=tf.float32, name="%s_eps_%d" % (name, i))
                self.gaussian_inputs.append(eps)
                pre_transform = eps * latent["q_stddev"] + latent["q_mean"]

                transform = latent["transform"]
                if transform is not None:
                    node, log_jacobian = transform(pre_transform)
                    stochastic_elbo_terms.append(log_jacobian)
                else:
                    node = pre_transform
                symbols[name] = node

                if "samples" not in latent:
                    latent["samples"] = []
                latent["samples"].append(node)

                
            symbols.update(self.jd_args)
            joint_density = self.joint_density(**symbols)
            stochastic_elbo_terms.append(joint_density)

        self.elbo = self.total_entropy + 1.0/n_eps * tf.add_n(stochastic_elbo_terms)
        return self.elbo
    
    def sample_stochastic_inputs(self, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
            
        for eps in self.gaussian_inputs:
            feed_dict[eps] = np.random.randn()

        return feed_dict
            
    def get_posterior_samples(self, latent_name):
        return tf.pack(self.latents[latent_name]["samples"])
