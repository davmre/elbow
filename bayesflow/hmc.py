import numpy as np
import tensorflow as tf
import bayesflow as bf
import bayesflow.util as util

class HMCInferenceFunctional(object):

    """
    HMC implemented in a functional style: each leapfrog step creates
    a new set of variables whose values depend on the state from the previous
    step. This creates a large graph but allows the entire update to be 
    encapsulated in a single TF op. 

    The alternative, yet to be implemented, is be to use external (Python)
    code to drive the leapfrog updates in-place. Not yet sure what the
    tradeoffs are here.

    """
    
    def __init__(self, joint_density, **jd_args):
        self.joint_density = joint_density
        self.jd_args = jd_args

        self.latents = {}
        self.symbols_gold = {}
        
        self.gaussian_inputs = []

        
    def add_latent(self, name, initial):
            
        with tf.name_scope("latent_" + name) as scope:
            latent = {}
            gold_var = tf.Variable(initial, name=name)
            self.symbols_gold[name] = gold_var
            
            latent["gold"] = gold_var
            latent["path"] = []            
            latent["path"].append(gold_var)
            latent["grad"] = []
            latent["momentum"] = []
            latent["samples"] = []

            shape = util._tf_extract_shape(gold_var)
            latent["shape"] = shape

            init_p = tf.placeholder(dtype=tf.float32,
                                    shape=shape,
                                    name="init_p")
            self.gaussian_inputs.append(init_p)
            latent["init_p"] = init_p

        self.latents[name] = latent

    def build_hmc_update(self, L=0, eps=1e-2):
        with tf.name_scope("gold_model"):
            self.joint_density_gold = self.joint_density(**self.symbols_gold)

            for latent in self.latents.values():
                grad = tf.gradients(self.joint_density_gold, latent["gold"])[0]
                momentum = latent["init_p"] + eps * grad / 2.0
                latent["grad"].append(grad)
                latent["momentum"].append(momentum)

        lps = []
        for i in range(L+1):
            with tf.name_scope("leapfrog_%d" % i):
                new_symbols = {}
                for name, latent in self.latents.items():                
                    old_x = latent["path"][i]
                    new_x = old_x + eps * latent["momentum"][i]
                    latent["path"].append(new_x)
                    new_symbols[name] = new_x
                    
                lp = self.joint_density(**new_symbols)
                lps.append(lp)
                for latent in self.latents.values():
                    grad = tf.gradients(lp, latent["path"][i+1])[0]
                    if i != L:
                        momentum = latent["momentum"][i] + eps * grad
                    else:
                        momentum = latent["momentum"][i] + eps * grad / 2.0
                    latent["grad"].append(grad)
                    latent["momentum"].append(momentum)

        for latent in self.latents.values():
            init_p = latent["init_p"]
            final_p = latent["momentum"][-1]                            
            latent["init_K"] = tf.reduce_sum(init_p*init_p)/2.0
            latent["final_K"] = tf.reduce_sum(final_p*final_p)/2.0

        with tf.name_scope("mh_updates"):
            init_Ks = tf.pack([latent["init_K"] for latent in self.latents.values()])
            self.init_K = tf.reduce_sum(init_Ks)

            final_Ks = tf.pack([latent["final_K"] for latent in self.latents.values()])
            self.final_K = tf.reduce_sum(final_Ks)

            self.final_density = lps[-1]
            self.mh_ratio = (self.final_density - self.final_K)  - (self.joint_density_gold - self.init_K)
            self.uniform = tf.placeholder(dtype=tf.float32, name="u")
            log_uniform = tf.log(self.uniform)
            self.accepted = log_uniform < self.mh_ratio 
            
            update_ops = []
            for name, latent in self.latents.items():

                if len(latent["shape"]) > 0:
                    accepted_vec = tf.tile(tf.reshape(self.accepted, (1,)), latent["shape"]) # HACK
                else:
                    accepted_vec = self.accepted
                gold_x = latent["gold"]
                proposed_x = latent["path"][-1]
                next_x = tf.select(accepted_vec, proposed_x, gold_x)
                update_ops.append(gold_x.assign(next_x))

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
            
    def train(self, steps=10000, print_interval=50,
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
        


