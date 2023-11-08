import numpy as np
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from parser import parser
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix


# TODO: YOUR CODE HERE
# Play with diffrent variations of parameters in your experiments
n_states = 2  # the number of HMM states
n_mixtures = 2  # the number of Gaussians
gmm = True  # whether to use GMM or plain Gaussian
covariance_type = "diag"  # Use diagonal covariange


# Gather data separately for each digit
def gather_in_dic(X, labels, spk):
    dic = {}
    for dig in set(labels):
        x = [X[i] for i in range(len(labels)) if labels[i] == dig]
        lengths = [len(i) for i in x]
        y = [dig for _ in range(len(x))]
        s = [spk[i] for i in range(len(labels)) if labels[i] == dig]
        dic[dig] = (x, lengths, y, s)
    return dic


def create_data():
    X, X_test, y, y_test, spk, spk_test = parser("recordings", n_mfcc=13)

    # TODO: YOUR CODE HERE
    (
        X_train,
        X_val,
        y_train,
        y_val,
        spk_train,
        spk_val,
    ) = ...  # split X into a 80/20 train validation split
    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))

    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels


def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):
    # TODO: YOUR CODE HERE
    dists = []
    for _ in range(n_states):
        distributions = ...  # n_mixtures gaussian distributions
        a = GeneralMixtureModel(distributions, verbose=True).fit(
            np.concatenate(X)
        )  # Concatenate all frames from all samples into a large matrix
        dists.append(a)
    return dists


def initialize_and_fit_normal_distributions(X, n_states):
    dists = []
    for _ in range(n_states):
        # TODO: YOUR CODE HERE
        d = ...  # Fit a normal distribution on X
        dists.append(d)
    return dists


def initialize_transition_matrix():
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    return ...


def initialize_starting_probabilities():
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    return ...


def initialize_end_probabilities():
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    return ...


def train_single_hmm(X, emission_model, digit):
    A = initialize_transition_matrix()
    start_probs = initialize_starting_probabilities()
    end_probs = initialize_end_probabilities()
    data = [x.astype(np.float32) for x in X]

    model = DenseHMM(
        distributions=emission_model,
        edges=A,
        starts=start_probs,
        ends=end_probs,
        verbose=True,
    ).fit(data)
    return model


def train_hmms(train_dic, labels):
    hmms = {}  # create one hmm for each digit

    for dig in labels:
        X, _, _, _ = train_dic[dig]
        # TODO: YOUR CODE HERE
        emission_model = ...
        hmms[dig] = ...
    return hmms


def evaluate(hmms, dic, labels):
    pred, true = [], []
    for dig in labels:
        X, _, _, _ = dic[dig]
        for sample in X:
            ev = []
            for _ in labels:
                sample = np.expand_dims(sample, 0)
                # TODO: YOUR CODE HERE
                logp = ...  # use the hmm.log_probability function
                ev.append(logp)

            # TODO: YOUR CODE HERE
            predicted_digit = ...  # Calculate the most probable digit
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true


train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
hmms = train_hmms(train_dic, labels)


labels = list(set(y_train))
pred_val, true_val = evaluate(hmms, val_dic, labels)

pred_test, true_test = evaluate(hmms, test_dic, labels)


# TODO: YOUR CODE HERE
# Calculate and print the accuracy score on the validation and the test sets
# Plot the confusion matrix for the validation and the test set
