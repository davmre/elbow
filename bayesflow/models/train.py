import numpy as np
import tensorflow as tf

def construct_elbo(*evidence_nodes):
    """
    Given a set of evidence nodes X, construct the evidence lower bound ELBO <= log p(X)
    integrating over all other nodes in the model. 

    Returns:
      elbo: TF scalar representing the (stochastic) ELBO
      sample_stochastic_inputs: a function that, when run, samples a feed_dict consisting of
                                all the stochastic inputs needed for a single-sample Monte Carlo 
                                approximation to the ELBO.
      decompose_elbo: a convenience function that, given a TF session and feed_dict, generates
                      a dictionary mapping variable names to the values of their corresponding
                      terms in the ELBO
      inspect_posterior: a convenience function that, given a TF session and feed_dict, generates
                      a dictionary mapping variable names to the parameter values of their
                      approximating Q distributions.                     
    """

    model_nodes = set([ancestor for node in evidence_nodes for ancestor in node.ancestors])
    node_terms = {}
    node_posteriors = {}
    
    expected_likelihoods = []
    entropies = []

    for node in model_nodes:
        elogp, entropy = node.elbo_term()
        node_terms[node.name] = (elogp, entropy)
        
        expected_likelihoods.append(elogp)
        entropies.append(entropy)
        
        qdist = node.q_distribution
        node_posteriors[node.name] = qdist.params()
        
    expected_likelihood = tf.reduce_sum(tf.pack(expected_likelihoods))
    entropy = tf.reduce_sum(tf.pack(entropies))
    elbo = expected_likelihood + entropy
    

    def decompose_elbo(session, feed_dict):
        terms = {}
        for name, (elogp, entropy) in node_terms.items():
            terms[name] = session.run((elogp, entropy), feed_dict=feed_dict)
        return terms

    def inspect_posterior(session, feed_dict):
        variables = {}
        for name, params in node_posteriors.items():
            variables[name] = {}
            for param_name, param_tensor in params.items():
                variables[name][param_name] = session.run(param_tensor, feed_dict=feed_dict)
        return variables

    q_distributions = set([node.q_distribution for node in model_nodes])
    def sample_stochastic_inputs():
        return {var: val for qdist in q_distributions for (var, val) in qdist.sample_stochastic_inputs().items()}

    return elbo, sample_stochastic_inputs, decompose_elbo, inspect_posterior

def optimize_elbo(node, steps=200, adam_rate=0.1, debug=False, return_session=False):
    """
    Convenience function to optimize an ELBO and return the breakdown of the final bound as well
    as the estimated posterior. 
    """
    
    elbo, sample_stochastic, decompose_elbo, inspect_posterior = construct_elbo(node)

    try:
        train_step = tf.train.AdamOptimizer(adam_rate).minimize(-elbo)
    except ValueError as e:
        print e
        steps = 0
                                
    init = tf.initialize_all_variables()

    if debug:
        debug_ops = tf.add_check_numerics_ops()

    
    sess = tf.Session()
    sess.run(init)
    for i in range(steps):
        fd = sample_stochastic()

        if debug:
            sess.run(debug_ops, feed_dict = fd)

        sess.run(train_step, feed_dict = fd)
        
        elbo_val = sess.run((elbo), feed_dict=fd)
        print i, elbo_val

        
    fd = sample_stochastic()    
    elbo_terms = decompose_elbo(sess, fd)
    posterior = inspect_posterior(sess, fd)

    if return_session:
        return elbo_terms, posterior, sess, fd
    else:
        sess.close()
        return elbo_terms, posterior

def print_inference_summary(elbo_terms, posterior):
    """
    Convenience function to print a summary of inference
    results, including the contribution of each term
    in the final bound. 
    """
    for k in sorted(elbo_terms.keys()):
        if "fixed" in k: continue
        elogp, entropy = elbo_terms[k]
        params = posterior[k]
        print k, "logp", elogp, "entropy", entropy
        for param, pval in params.items():
            if isinstance(pval, np.ndarray) and pval.size > 3: continue
            print "  ", param, pval
