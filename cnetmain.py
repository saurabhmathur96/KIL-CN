# import pyximport; pyximport.install(language_level = 3)
from leaf import fit_base, Dataset, fit_base, fit_grad
from cnet import learn_cnet, score
import pandas as pd
import numpy as np
from dataset import get_dataset

np.random.seed(0)
X_train, X_test, r, C = get_dataset("abalone")
X_train = X_train
print (len(X_train))
alpha, epsilon, lambda_ = 1, 0.001, 0
min_instances = 500
n = len(C)
scope = list(range(n))
train = Dataset(X_train, r)
test = Dataset(X_test, r)
cnet = learn_cnet(train, scope, alpha, C, epsilon, lambda_, min_instances)
print (f"{cnet.loglik(train.X):+.8f}, {cnet.loglik(test.X):+.8f}, {cnet.penalty(C, epsilon):+.8f}")