import numpy as np
import tensorflow as tf
import bayesflow as bf
import scipy.stats

"""

Implement a toy Bernoulli model with a Beta approximate posterior. Note that by conjugacy the true posterior is in fact also Beta, and obtainable analytically, but we'll pretend we don't know that and estimate it by black box VI instead. In particular we'll use the score function gradient estimate. (TODO: is it possible to directly compare to the reparameterization trick?)
"""


"""
how does the score fn estimator work? I define the ELBO as 
E_q(z)[log p(x,z)] + entropy(q)
where I could / should? implement the analytic entropy for the beta distribution, though it's unclear what the right control variate strategy is here, since we have the choice of
E_q[log p(x,z) - log q(z)]
E_q[log p(x,z)] + entropy(q)
E_q[log p(x|z)] - kl(q|p)
this is an interesting thing to experiment on, it seems like there should be a RIGHT ANSWER even theoretically (analyzing the variance), but for now I'll do the first since it's simplest and doesn't require me to research the beta distribution. 

Okay so the SCG is that I'll have parameters A=(a,b) controlling the q distribution. Let f_A(x,z) = log p(x,z) - log q_A(z). Then 
grad_A int_z q_A(z) f_A(x,z)
= int_z grad [q(z) f(x,z)]
= int_z [grad q(z)] * f(x,z) + int_z q(z) grad log q(z)
= int_z q(z) [grad log q(z)] * f(x,z) + int_z q(z) / q(z) [grad q(z)]
= int_z q(z) [grad log q(z)] * f(x,z) + grad int_z q(z)
= int_z q(z) [grad log q(z)] * f(x,z)
which we approximate by drawing Z ~ q(z) and then computing 
sum_i [grad_A log q_A(Z_i)] * f(x,Z_i)]
as an approximation to the gradient. Now the question is, what is the deterministic fn that this is the gradient of? I guess it's just 
g(A) = log q_A(Z_i) * f(x, Z_i)
so the "natural" way to do inference would be to implement this as the TF expression over which I compute gradients. 
except wait that's not right because the gradient of this quantity ends up pushing inside the f and giving an extra term. like if I differentiated this I'd get
grad log q_A(Z_i) * f(x, Z_i) - log q_A(Z_i) grad log q_A(Z_a)
which is not right. is there an integration by parts waiting to happen here? Let me parse the SCG paper. 
what is the actual SCG here? I start with the params A. 
Then I sample z ~ q_A(z) at a stochastic node. 
Then I compute log p(x, z) and log q_A(z) at two deterministic nodes both depending on z
Then I subtract these to get a final cost L = log p(x,z) - log q_A(z)
so the graph is
A -> stochastic(z) -> log p(x,z)
 \    /                   |
  log q_A(z)      ->    L
and I am interested in the gradient of L given A. I could also write this more simply as
A -> stochastic(z)
 \   /
  L(x, z, A)

SCG gives gradient estimate as
E_q(Z) [ grad_A log q_A(Z) f(x, Z) + grad_A f(x, Z)]
= E_q(Z) [ grad_A log q_A(Z) f(x, Z) - grad_A log q_A(Z)]
which is equal to 
= E_q(Z) [ grad_A log q_A(Z) f(x, Z)]
analytically but the actual estimator will be different. 
SCG defines the surrogate loss as
log q_A(Z_i) * f(x, Z_i) + f(x, Z_i)
which when we differentiate will be
grad log q_A(Z_i) * f(x, Z_i) - log q_A(Z_i) grad log q_A(Z) - grad log q_A(Z)
except WAIT the scg paper treats the costs Q_w as fixed constants so the product rule doesn't apply and the "official" derivative is
grad log q_A(Z_i) * f(x, Z_i) - grad log q_A(Z)
but how do I encode this in the surrogate function? 
I mean one option is to have the "real" elbo graph, do the sampling, compute the relevant cost values, and then copy them as placeholders into the surrogate cost fn. But this is screamingly inelegant. 
  - I guess another option is to basically force this to happen inside of TF somehow, if it's possible to specify the surrogate fn in a way that depends on f but doesn't actually propagate gradients with respect to f, but I'd almost be disappointed in TF if there's a way to get that functionality. 



"""

class BernoulliModel(object):

    def __init__(self, N, n_thetas=1):

        self.N = N

        self.theta_q_alpha = tf.Variable(1.0, name="theta_q_alpha")
        self.theta_q_beta = tf.Variable(2.0, name="theta_q_beta")

        self.data = tf.placeholder(dtype=tf.float32, shape=(N,), name="data")

        self.thetas = tf.placeholder(shape=(n_thetas,), dtype=tf.float32, name="thetas")
        
        self.thetas_q_log_density = bf.dists.beta_log_density(self.thetas, alpha=self.theta_q_alpha, beta=self.theta_q_beta)
        self.thetas_prior = bf.dists.beta_log_density(self.thetas, alpha=1., beta=1.) 

        self.data_liks = tf.pack([tf.reduce_sum(bf.dists.bernoulli_log_density(self.data, theta)) for theta in tf.unpack(self.thetas)])
        self.joint_density = self.data_liks + self.thetas_prior
        
        self.stochastic_elbo = self.joint_density - self.thetas_q_log_density

        # TODO: add control variates
        self.surrogate = tf.reduce_mean(self.thetas_q_log_density * tf.stop_gradient(self.stochastic_elbo) + self.stochastic_elbo)




bernoulli_data = (0,1,0,0,0,0,0,0,0,1)
N = len(bernoulli_data)

n_thetas = 100
model = BernoulliModel(N, n_thetas=n_thetas)
    
train_step = tf.train.AdamOptimizer(0.1).minimize(-model.surrogate)
init = tf.initialize_all_variables()
        
sess = tf.Session()
sess.run(init)

for i in range(10000):
    feed_dict = {model.data: bernoulli_data}

    alpha, beta = sess.run([model.theta_q_alpha, model.theta_q_beta])
    
    rv = scipy.stats.beta(alpha, beta)
    thetas = rv.rvs(n_thetas)
    feed_dict[model.thetas] = thetas

    (q, prior, lik, joint, elbos, surrogate) = sess.run([model.thetas_q_log_density, model.thetas_prior, model.data_liks, model.joint_density, model.stochastic_elbo, model.surrogate, ], feed_dict=feed_dict)

    if i % 100 == 0:
        print "step %d alpha %.3f beta %.3f theta %.3f q_density %.2f prior %.2f lik %.2f joint %.2f  elbo %.2f surrogate %.2f " % (i, alpha, beta, np.mean(thetas), np.mean(q), np.mean(prior), np.mean(lik), np.mean(joint), np.mean(elbos), surrogate)

    sess.run(train_step, feed_dict = feed_dict)
        
