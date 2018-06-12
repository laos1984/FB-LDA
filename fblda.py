"""
(C) Shulong Tan - 2018

Implementation of the collapsed Gibbs sampler for
Foreground and Background Latent Dirichlet Allocation (FB-LDA), as described in
"Interpreting the Public Sentiment Variations on Twitter". Shulong Tan et al. TKDE 2014.
"""

import numpy as np
import scipy as sp
import scipy.misc
from scipy.special import gammaln

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)

class FBLDASampler(object):

    def __init__(self, n_f_topics, n_b_topics, alpha_theta=0.1, alpha_lambda=0.5, alpha_mu=0.1, beta_f=0.01, beta_b=0.01):

        """
        n_f_topics: desired number of foreground topics
        n_b_topics: desired number of background topics

        """
        self.n_f_topics = n_f_topics
        self.n_b_topics = n_b_topics
        self.alpha_theta = alpha_theta
        self.alpha_lambda = alpha_lambda
        self.alpha_mu = alpha_mu
        self.beta_f = beta_f
        self.beta_b = beta_b

    def _initialize(self, matrix_f, matrix_b):
        n_f_docs, vocab_size = matrix_f.shape
        n_b_docs, vocab_size= matrix_b.shape

        # number of times background document n_b_docs and background topic n_b_topics co-occur
        self.Psi_bb=np.zeros((n_b_docs,self.n_b_topics))
        self.Psi_b=np.zeros(n_b_docs)

        # number of times foreground document n_f_docs and background topic n_b_topics co-occur
        self.Psi_fb=np.zeros((n_f_docs,self.n_b_topics))
        self.Psi_f=np.zeros(n_f_docs) 

        # number of times foreground document n_f_docs and foreground topic n_f_topics co-occur
        self.Omega_ff=np.zeros((n_f_docs,self.n_f_topics))
        self.Omega_f=np.zeros(n_f_docs)

        # number of times foreground topic f and word w co-occur
        self.Theta_vf = np.zeros((self.n_f_topics, vocab_size))
        self.Theta_f = np.zeros(self.n_f_topics)

        # number of times background topic f and word w co-occur
        self.Delta_vb = np.zeros((self.n_b_topics, vocab_size))
        self.Delta_b = np.zeros(self.n_b_topics)

        self.Mf=np.zeros((n_f_docs, 2)) #0 for foreground topics and 1 for background topics

        self.topics_ff = {}
        self.topics_fb = {}
        self.topics_b = {}
        self.topics_m = {}

        for m in xrange(n_b_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix_b[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_b_topics)
                self.Psi_bb[m,z] += 1
                self.Psi_b[m] += 1
                self.Delta_vb[z,w] += 1
                self.Delta_b[z] += 1
                self.topics_b[(m,i)] = z

        for m in xrange(n_f_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix_f[m, :])):
                # choose an arbitrary topic type as topic type for word i
                y = np.random.randint(2)
                self.Mf[m,y] += 1
                self.topics_m[(m,i)]=y
                if y==0: # background topic
                    # choose an arbitrary topic as first topic for word i
                    z = np.random.randint(self.n_f_topics)
                    self.Omega_ff[m,z] += 1
                    self.Omega_f[m] += 1
                    self.Theta_vf[z,w] += 1
                    self.Theta_f[z] += 1
                    self.topics_ff[(m,i)] = z
                    
                else: #foreground topic
                    # choose an arbitrary topic as first topic for word i
                    z = np.random.randint(self.n_b_topics)
                    self.Psi_fb[m,z] += 1
                    self.Psi_f[m] += 1
                    self.Delta_vb[z,w] += 1
                    self.Delta_b[z] += 1
                    self.topics_fb[(m,i)] = z
               
    def _conditional_distribution_b(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.Delta_vb.shape[1]
        left = (self.Delta_vb[:,w] + self.beta_b) / \
               (self.Delta_b + self.beta_b * vocab_size)
        right = (self.Psi_bb[m,:] + self.alpha_mu) / \
                (self.Psi_b[m] + self.alpha_mu * self.n_b_topics)
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def _conditional_distribution_f(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """
        p_z_c=np.zeros(self.n_f_topics+self.n_b_topics)
        vocab_size = self.Delta_vb.shape[1]
        left = (self.Theta_vf[:,w] + self.beta_f) / \
               (self.Theta_f + self.beta_f * vocab_size)
        middle=(self.Omega_ff[m,:] + self.alpha_theta) / \
                (self.Omega_f[m] + self.alpha_theta * self.n_f_topics)
        right = (self.Mf[m,0] + self.alpha_lambda) / \
                (self.Mf[m,0]+ self.Mf[m,1]+ self.alpha_lambda * 2)
        p_z = left * middle * right
        #print np.sum(p_z)
        # normalize to obtain probabilities
        #p_z /= np.sum(p_z)
        p_z_c[0:self.n_f_topics]=p_z


        left = (self.Delta_vb[:,w] + self.beta_b) / \
               (self.Delta_b + self.beta_b * vocab_size)
        middle=(self.Psi_fb[m,:] + self.alpha_mu) / \
                (self.Psi_f[m] + self.alpha_mu * self.n_b_topics)
        right = (self.Mf[m,1] + self.alpha_lambda) / \
                (self.Mf[m,0]+ self.Mf[m,1] + self.alpha_lambda * 2)
        p_z = left * middle * right
        # normalize to obtain probabilities
        #p_z /= np.sum(p_z)
        p_z_c[self.n_f_topics:self.n_f_topics+self.n_b_topics]=p_z
        p_z_c /= np.sum(p_z_c)

        return p_z_c


    # def loglikelihood(self):
    #     """
    #     Compute the likelihood that the model generated the data.
    #     """
    #     vocab_size = self.nzw.shape[1]
    #     n_docs = self.nmz.shape[0]
    #     lik = 0

    #     for z in xrange(self.n_topics):
    #         lik += log_multi_beta(self.nzw[z,:]+self.beta)
    #         lik -= log_multi_beta(self.beta, vocab_size)

    #     for m in xrange(n_docs):
    #         lik += log_multi_beta(self.nmz[m,:]+self.alpha)
    #         lik -= log_multi_beta(self.alpha, self.n_topics)

    #     return lik

    def phi_f(self):
        """
        Compute phi_f = p(w|z_f).
        """
        V = self.Theta_vf.shape[1]
        num = self.Theta_vf + self.beta_f
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num


    def phi_b(self):
        """
        Compute phi_f = p(w|z_f).
        """
        #V = self.Delta_vb.shape[1]
        num = self.Delta_vb + self.beta_b
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    # def phi_b(self):
    #     """
    #     Compute phi_f = p(w|z_f).
    #     """
    #     vocab_size = self.Theta_vf.shape[1]
    #     num = (self.Theta_vf[:,:] + self.beta_f) / \
    #            (self.Theta_f + self.beta_f * vocab_size)
    #     return num

    def theta_f(self):
        """
        Compute phi_f = p(w|z_f).
        """
        #V = self.Omega_ff.shape[1]
        num = self.Omega_ff + self.alpha_theta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def lambda_f(self):
        """
        Compute phi_f = p(w|z_f).
        """
        #V = self.Mf.shape[1]
        num = self.Mf + self.alpha_lambda
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num
        
    def mu_f(self):
        """
        Compute phi_f = p(w|z_f).
        """
        #V = self.Psi_fb.shape[1]
        num = self.Psi_fb + self.alpha_mu
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num
        

    def mu_b(self):
        """
        Compute phi_f = p(w|z_f).
        """
        #V = self.Psi_bb.shape[1]
        num = self.Psi_bb + self.alpha_mu
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def run(self, matrix_f, matrix_b, maxiter=100):
        """
        Run the Gibbs sampler.
        """
        n_f_docs, vocab_size = matrix_f.shape
        n_b_docs, vocab_size= matrix_b.shape
        self._initialize(matrix_f,matrix_b)

        for it in xrange(maxiter):
            for m in xrange(n_b_docs):
                for i, w in enumerate(word_indices(matrix_b[m, :])):
                    z = self.topics_b[(m,i)]

                    self.Psi_bb[m,z] -= 1
                    self.Psi_b[m] -= 1
                    self.Delta_vb[z,w] -= 1
                    self.Delta_b[z] -= 1

                    p_z = self._conditional_distribution_b(m, w)
                    z = sample_index(p_z)

                    self.Psi_bb[m,z] += 1
                    self.Psi_b[m] += 1
                    self.Delta_vb[z,w] += 1
                    self.Delta_b[z] += 1
                    self.topics_b[(m,i)] = z

            for m in xrange(n_f_docs):
                for i, w in enumerate(word_indices(matrix_f[m, :])):
                    y = self.topics_m[(m,i)]
                    # self.Mf[m,y] -= 1
                    # p_y = self._conditional_distribution(m, w)
                    # y = sample_index(p_y)
                    # self.Mf[m,y] += y

                    if y==0:
                        self.Mf[m,0] -= 1
                        z = self.topics_ff[(m,i)]
                        self.Omega_ff[m,z] -= 1
                        self.Omega_f[m] -= 1
                        self.Theta_vf[z,w] -= 1
                        self.Theta_f[z] -= 1
                        
                    else:
                        self.Mf[m,1] -= 1
                        z = self.topics_fb[(m,i)]
                        self.Psi_fb[m,z] -= 1
                        self.Psi_f[m] -= 1
                        self.Delta_vb[z,w] -= 1
                        self.Delta_b[z] -= 1

                    p_z = self._conditional_distribution_f(m, w)
                    #print p_z
                    z = sample_index(p_z)
                    if z<self.n_f_topics:
                        self.Mf[m,0] += 1
                        self.Omega_ff[m,z] += 1
                        self.Omega_f[m] += 1
                        self.Theta_vf[z,w] += 1
                        self.Theta_f[z] += 1
                        self.topics_m[(m,i)] = 0 
                        self.topics_ff[(m,i)] = z
                    else:
                        z=z-self.n_f_topics
                        self.Mf[m,1] += 1
                        self.Psi_fb[m,z] += 1
                        self.Psi_f[m] += 1
                        self.Delta_vb[z,w] += 1
                        self.Delta_b[z] += 1
                        self.topics_m[(m,i)] = 1 
                        self.topics_fb[(m,i)] = z
            # FIXME: burn-in and lag!
            yield (self.phi_f(), self.phi_b())