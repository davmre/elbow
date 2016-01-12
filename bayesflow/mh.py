import numpy as np
import tensorflow as tf
import bayesflow as bf
import bayesflow.util as util

class RandomWalkMHInference(object):

    def __init__(self, joint_density, **jd_args):
        self.joint_density = joint_density
        self.jd_args = jd_args

        self.latents = {}

        self.gaussian_inputs = []

        self.symbols_gold = {}
        self.symbols_proposed = {}
        
    def add_latent(self, name, initial, step_size=1.0):
            
        with tf.name_scope("latent_" + name) as scope:
            latent = {}
            latent["step_size"] = step_size
            latent["gold"] = tf.Variable(initial, name=name)
            latent["samples"] = []
            self.symbols_gold[name] = latent["gold"]

            shape = util._tf_extract_shape(latent["gold"])
            eps = tf.placeholder(dtype=tf.float32,
                                 shape=shape,
                                 name="eps")
            self.gaussian_inputs.append(eps)
            latent["proposed"] = eps * latent["step_size"] + latent["gold"]
            self.symbols_proposed[name] = latent["proposed"]
            
        self.latents[name] = latent

    def build_mh_update(self):
        with tf.name_scope("gold_model"):
            self.joint_density_gold = self.joint_density(**self.symbols_gold)

        with tf.name_scope("proposed_model"):
            self.joint_density_proposed = self.joint_density(**self.symbols_proposed)
        with tf.name_scope("mh_updates"):            
            self.mh_ratio = self.joint_density_proposed - self.joint_density_gold
            self.uniform = tf.placeholder(dtype=tf.float32, name="u")
            log_uniform = tf.log(self.uniform)
            self.accepted = log_uniform < self.mh_ratio 
            
            update_ops = []
            for name, latent in self.latents.items():
                next_val = tf.select(self.accepted, latent["proposed"], latent["gold"])
                update_ops.append(latent["gold"].assign(next_val))

            self.step_counter = tf.Variable(0)
            self.accept_counter = tf.Variable(0)
            self.accept_rate = tf.to_double(self.accept_counter) / tf.to_double(self.step_counter)
            update_ops.append(self.step_counter.assign_add(1))
            update_ops.append(self.accept_counter.assign_add(tf.select(self.accepted, 1, 0)))
            
            self.global_update = tf.group(*update_ops)
                
        return self.global_update
    
    def sample_stochastic_inputs(self, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}

        feed_dict[self.uniform] = np.random.rand()
            
        for eps in self.gaussian_inputs:
            shape = util._tf_extract_shape(eps)
            feed_dict[eps] = np.random.randn(*shape)

        return feed_dict
            
    def train(self,
              steps=10000, print_interval=50,
              logdir=None,
              display_dict=None, sess=None):
        
        if display_dict is None or len(display_dict)==0:
            print_names = []
            print_vars = []
        else:
            print_names, print_vars = zip(*display_dict.items())
        print_names = ["lik", "accepted"] + list(print_names)
        print_vars = [self.joint_density_gold,self.accept_rate] + list(print_vars)

        train_step = self.global_update
        init = tf.initialize_all_variables()

        if sess is None:
            sess = tf.Session()
            
        if logdir is not None:
            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)

        names, dicts = zip(*self.latents.items())
        gold_vars = [d["gold"] for d in dicts]
        
        sess.run(init)
        for i in range(steps):
            fd = self.sample_stochastic_inputs()
            
            if i % print_interval == 0:
                print_vals  = sess.run(print_vars, feed_dict=fd)
                print_str = " ".join(["%s %.4f" % (n, v) for (n, v) in zip(print_names, print_vals)])
                print ("step %d " % i) + print_str

                if logdir is not None:
                    summary_str = sess.run(merged, feed_dict=fd)
                    writer.add_summary(summary_str, i)

            sess.run(train_step, feed_dict = fd)

            vals = sess.run(gold_vars, feed_dict = fd)
            for name, val in zip(names, vals):
                self.latents[name]["samples"].append(val)
        
