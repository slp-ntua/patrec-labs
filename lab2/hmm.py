import numpy as np
from pomegranate import *

X = [] # data from a single digit (can be a numpy array)

n_states = 2 # the number of HMM states
n_mixtures = 2 # the number of Gaussians
gmm = True # whether to use GMM or plain Gaussian

dists = [] # list of probability distributions for the HMM states
for i in range(n_states):
    if gmm:
        a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, num_mixtures, X)
    else:
        a = MultivariateGaussianDistribution.from_samples(X)
    dists.append(a)

trans_mat = [] # your transition matrix
starts = [] # your starting probability matrix
ends = [] # your ending probability matrix

data = [] # your data: must be a Python list that contains: 2D lists with the sequences (so its dimension would be num_sequences x seq_length x feature_dimension)
          # But be careful, it is not a numpy array, it is a Python list (so each sequence can have different length)

# Define the GMM-HMM
model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])

# Fit the model
model.fit(data, max_iterations=5)

# Predict a sequence
sample = [] # a sample sequence
logp, _ = model.viterbi(sample) # Run viterbi algorithm and return log-probability