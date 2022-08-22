from typing import *

from itertools import product
import numpy as np
from scipy.special import logsumexp
from leaf2 import Dataset, ChowLiuTree, fit_grad


class Node:
  scope: list

  def __init__(self, scope):
    self.scope = scope

  def __repr__(self):
    return f"<{self.__class__.__name__} scope={self.scope}>"


class Leaf(Node):
  C: np.array
  epsilon: float
  lambda_: float
  alpha: float
  r: list
  clt: ChowLiuTree

  def __init__(self, scope, r, C, epsilon, lambda_, alpha):
    super().__init__(scope)

    self.r = [r[s] for s in scope]
    C2 = np.zeros((len(scope), len(scope)))
    for (i, si), (j, sj) in product(enumerate(scope), enumerate(scope)):
      C2[i, j] = C[si, sj]

    self.C = C2
    self.lambda_ = lambda_
    self.alpha = alpha
    self.epsilon = epsilon

  def fit(self, X):
    D = Dataset(X[:, self.scope], self.r)
    self.clt = fit_grad(D, self.C, self.epsilon, self.lambda_, self.alpha)

    return self

  def loglik(self, X):
    return self.clt.loglik(X[:, self.scope])

  def logmar(self, query):
    q = [(self.scope.index(i), vi) for (i, vi) in query]
    return self.clt.logmar(q)

  def penalty(self, C, epsilon) -> float:
    scope = self.scope
    C2 = np.zeros((len(scope), len(scope)))
    for (i, si), (j, sj) in product(enumerate(scope), enumerate(scope)):
      C2[i, j] = C[si, sj]
    return self.clt.penalty(C2, epsilon)


class InternalNode(Node):
  i: int
  r: list
  values: list[int]
  weights: np.array
  children: list[ChowLiuTree]

  def __init__(self, i, scope, r, values, weights, children):
    super().__init__(scope)
    self.i = i  # splitting attribute
    self.r = r
    self.values = values  # value corresponding to each child
    self.weights = weights
    self.children = children

  def loglik(self, X):
    j = self.i
    ll = np.array([child.loglik(X[X[:, j] == value])
                   for child, value in zip(self.children, self.values)])
    return logsumexp(ll, b=self.weights)

  def logmar(self, query):
    if self.i in [i for i, vi in query]:
      query2 = [(i, vi) for i, vi in query if i != self.i]
      for i, vi in query:
        if i == self.i:
          return np.log(self.weights[vi] * np.exp(self.children[vi].logmar(query2)))
    else:
      pq = np.array([c.logmar(query) for c in self.children])
      return logsumexp(pq + np.log(self.weights))

  def conditional(self, va, b, vb):
    # P(A | B)
    A = (self.i, va)
    B = (b, vb)
    return np.exp(self.logmar([A, B]) - self.logmar([B]))

  def penalty(self, C, epsilon) -> float:
    scope = self.scope
    children = self.children
    j = self.i
    jj = scope.index(j)  # local index
    penalty = 0
    for ii, i in enumerate(scope):
      if i == j or C[i, j] == 0:
        continue
      # i is monotonically influenced by j

      terms = np.array([
        np.cumsum([np.exp(children[vj].logmar([(i, vi)])) for vi in range(self.r[ii])])
        for vj in range(self.r[jj])]).T  # |j| x |i|, P(xi <= vi | xj = vj)
      rows = terms[:-1]
      delta = np.array([
        C[i, j] * (row[vj2] - row[vj1]) + epsilon
        for row in rows
        for vj2, vj1 in product(range(self.r[jj]), range(self.r[jj]))
        if vj2 > vj1
      ])
      penalty += np.sum((delta > 0).astype(int) * (delta ** 2))

    for ii, i in enumerate(scope):
      if i == j or C[j, i] == 0:
        continue
      # j is monotonically influenced by i
      # |i| x |j|
      terms = []
      for vi in range(self.r[ii]):
        marg = np.exp([children[vj].logmar([(i, vi)]) for vj in range(self.r[jj])])
        numer = marg * self.weights
        term = numer / np.sum(numer)
        terms.append(np.cumsum(term))
      terms = np.array(terms).T
      rows = terms[:-1]
      delta = np.array([
        C[j, i] * (row[vi2] - row[vi1]) + epsilon
        for row in rows
        for vi2, vi1 in product(range(self.r[ii]), range(self.r[ii]))
        if vi2 > vi1
      ])
      penalty += np.sum((delta > 0).astype(int) * (delta ** 2))

    for child in self.children:
      penalty += child.penalty(C, epsilon)

    return penalty

  def __repr__(self):
    return f"<InternalNode scope={', '.join([f'[{j}]' if self.i == j else f'{j}' for j in self.scope])}>"


def remove(ls: list, i: int):
  return [j for j in ls if j != i]


def split_data(X: np.ndarray, r: list, i: int):
  splits = []
  for v in range(r[i]):
    splits.append(X[X[:, i] == v])
  return splits



def learn_cnet(D: Dataset, scope: list, alpha: float, C: np.ndarray, epsilon: float, lambda_: float,
               min_instances: int):
  print(f"learn_cnet {len(D.X)}, {len(scope)}")
  if len(D.X) <= min_instances or len(scope) <= 3:
    return Leaf(scope, D.r, C, epsilon, lambda_, alpha).fit(D.X)

  candidates = [split_data(D.X, D.r, j) for j in scope]

  args = (C, epsilon, lambda_, alpha)
  scores = [score(j, c, D.r, remove(scope, j), *args) for c, j in zip(candidates, scope)]
  i = np.argmax(scores)
  splits = candidates[i]

  scope2 = remove(scope, scope[i])
  args = (scope2, alpha, C, epsilon, lambda_, min_instances)
  return InternalNode(scope[i],
                      scope,
                      D.r,
                      list(range(D.r[scope[i]])),
                      weights=np.array([len(c) for c in splits]) / len(D.X),
                      children=[learn_cnet(Dataset(c, D.r), *args) for c in splits])


def score(j: int, Xs: list, r: list, scope: list, C: np.ndarray, epsilon: float, lambda_: float, alpha: float):
  weights = np.array([len(X) for X in Xs], dtype=float)
  weights /= len(weights)
  # scope, r,  C, epsilon, lambda_, alpha, method, options
  leaves = [Leaf(scope, r, C, epsilon, lambda_, alpha).fit(X) for X in Xs]
  node = InternalNode(j, [j] + scope, r, list(range(r[j])), weights, leaves)
  return node.loglik(np.concatenate(Xs, axis=0)) + lambda_ * node.penalty(C, epsilon)