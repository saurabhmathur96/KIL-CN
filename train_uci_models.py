import numpy as np
from cnet2 import learn_cnet
from leaf2 import Dataset
from sklearn.base import BaseEstimator


class CNetEstimator(BaseEstimator):
  def __init__(self, r, alpha=1):
    self.r = r
    self.alpha = alpha

  def fit(self, X, *args):
    n = len(self.r)
    if n > 5:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha)
    else:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha, min_variables=n - 1)
    self.node = node
    return self

  def score(self, X):
    return self.node.loglik(X)


class CNetLocalSoftEstimator(BaseEstimator):
  def __init__(self, C, r, alpha=1, epsilon=0.1):
    self.r = r
    self.C = C
    self.alpha = alpha
    self.epsilon = epsilon

  def fit(self, X, *args):
    n = len(self.r)
    if n > 5:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha)
    else:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha, min_variables=n - 1)
    node.fit_leaves(X, self.alpha, self.C, self.epsilon, 10, 1)
    self.node = node
    return self

  def score(self, X):
    return self.node.loglik(X)


class CNetGlobalSoftEstimator(BaseEstimator):
  def __init__(self, C, r, alpha=1, epsilon=0.1):
    self.r = r
    self.C = C
    self.alpha = alpha
    self.epsilon = epsilon

  def fit(self, X, *args):
    n = len(self.r)
    if n > 5:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha)
    else:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha, min_variables=n - 1)
    node.fit(X, self.alpha, self.C, self.epsilon, 10, 1)
    self.node = node
    return self

  def score(self, X):
    return self.node.loglik(X)


class CNetLocalHardEstimator(BaseEstimator):
  def __init__(self, C, r, alpha=1, epsilon=0.1):
    self.r = r
    self.C = C
    self.alpha = alpha
    self.epsilon = epsilon

  def fit(self, X, *args):
    n = len(self.r)
    if n > 5:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha)
    else:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha, min_variables=n - 1)
    node.fit_leaves(X, self.alpha, self.C, self.epsilon, 1, 10)
    self.node = node
    return self

  def score(self, X):
    return self.node.loglik(X)


class CNetGlobalHardEstimator(BaseEstimator):
  def __init__(self, C, r, alpha=1, epsilon=0.1):
    self.r = r
    self.C = C
    self.alpha = alpha
    self.epsilon = epsilon

  def fit(self, X, *args):
    n = len(self.r)
    if n > 5:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha)
    else:
      node = learn_cnet(Dataset(X, self.r), list(range(n)), self.alpha, min_variables=n - 1)
    node.fit(X, self.alpha, self.C, self.epsilon, 1, 10)
    self.node = node
    return self

  def score(self, X):
    return self.node.loglik(X)