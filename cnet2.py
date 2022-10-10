from typing import *

from itertools import product
import numpy as np
from scipy.special import logsumexp, softmax, log_softmax
from scipy.optimize import check_grad, minimize
from leaf2 import *
import numpy.typing as npt
from joblib import Parallel, delayed


class Node:
  scope: List[int]

  def __init__(self, scope: List[int]):
    self.scope = scope

  def __repr__(self):
    return f"<{self.__class__.__name__} scope={self.scope}>"


class BaseLeaf(Node):
  r: List[int]
  scope: List[int]
  clt: ChowLiuTree

  def __init__(self, scope, r):
    super().__init__(scope)
    self.r = [r[s] for s in scope]

  def fit(self, X: npt.NDArray, alpha: float):
    D = Dataset(X[:, self.scope], self.r)
    parent = fit_structure(D, alpha)
    ss = sufficient_stats(parent, D)
    log_factors = fit_base_parameters(parent, D, alpha)
    self.clt = ChowLiuTree(parent, D.r, log_factors)
    return self

  def loglik(self, X: npt.NDArray):
    return self.clt.loglik(X[:, self.scope])

  def logmar(self, query: List[Tuple[int, int]]):
    q = [(self.scope.index(i), vi) for (i, vi) in query]
    return self.clt.logmar(q)

  def logp(self, X: npt.NDArray):
    return self.clt.logp(X[:, self.scope])

  def penalty(self, C: npt.NDArray, epsilon: float) -> float:
    scope = self.scope
    C2 = np.zeros((len(scope), len(scope)))
    for (i, si), (j, sj) in product(enumerate(scope), enumerate(scope)):
      C2[i, j] = C[si, sj]
    return self.clt.penalty(C2, epsilon)

  def delta(self, i: int, j: int, sign: int, epsilon: float):
    return self.clt.delta(self.scope.index(i), self.scope.index(j), sign, epsilon)

  @property
  def params_size(self):
    return self.clt.params_size


class Leaf(BaseLeaf):
  r: List[int]
  scope: List[int]
  clt: ChowLiuTree

  def __init__(self, scope: List[int], r: List[int]):
    super().__init__(scope, r)

  def fit(self, X: npt.NDArray, alpha: float, C: npt.NDArray, epsilon: float, lambda_: float):
    scope = self.scope
    D = Dataset(X[:, scope], self.r)
    C2 = np.zeros((len(scope), len(scope)))
    for (i, si), (j, sj) in product(enumerate(scope), enumerate(scope)):
      C2[i, j] = C[si, sj]

    self.clt = fit_grad(D, C2, epsilon, lambda_, alpha)
    self._theta = self.clt.log_factors
    return self

  def set_params(self, x: npt.NDArray):
    theta = unpack(x, self.clt.parent, self.r)
    log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1)
                   for t, p in zip(theta, self.clt.parent)]
    self.clt.log_factors = log_factors
    self._theta = theta

  @property
  def params(self):
    return pack(self._theta, self.clt.parent, self.r)

  def full_mar_grad(self, query: List[Tuple[int, int]]):
    theta = self.clt.log_factors
    factor_grads = [
      s_grad(t) if p < 0 else np.apply_along_axis(s_grad, 1, t)  # np.array([s_grad(t[vp]) for vp in range(r[p]) ])
      for t, p in zip(theta, self.clt.parent)
    ]
    q = [(self.scope.index(i), vi) for (i, vi) in query]
    g = mar_grad(self.clt, factor_grads, q)
    return pack(g, self.clt.parent, self.r)

  def loglik_grad(self, X: npt.NDArray, alpha: float):
    log_factors = self.clt.log_factors
    theta = self._theta
    factor_grads = [
      s_grad(t) if p < 0 else np.apply_along_axis(s_grad, 1, t)  # np.array([s_grad(t[vp]) for vp in range(r[p]) ])
      for t, p in zip(theta, self.clt.parent)
    ]
    reg_grad = [
      np.sum(factor_grads[i] * np.exp(-log_factors[i]), axis=1)
      if pi < 0
      else [np.sum(factor_grads[i][j] * np.exp(-log_factors[i][j]), axis=1) for j in range(self.clt.r[pi])]
      for i, pi in enumerate(self.clt.parent)
    ]
    ss = sufficient_stats(self.clt.parent, Dataset(X[:, self.scope], self.clt.r))
    ll_terms = loglik_grad(self.clt, ss)
    return pack(ll_terms, self.clt.parent, self.clt.r) + alpha * pack(reg_grad, self.clt.parent, self.clt.r)

  @property
  def reg(self):
    return sum([np.sum(e) for e in self.clt.log_factors])


class InternalNode(Node):
  i: int
  r: List[int]
  values: List[int]
  weights: npt.NDArray
  children: List[ChowLiuTree]

  def __init__(self, i: int, scope: List[int], r: List[int], values: npt.NDArray,
               weights: npt.NDArray, children: List[ChowLiuTree]):
    super().__init__(scope)
    self.i = i  # splitting attribute
    self.r = r
    self.values = values  # value corresponding to each child
    self.weights = weights
    self._theta = np.log(weights)
    self.children = children
    # self.params_size = len(self.weights) + sum(child.params_size for child in self.children)
    self.params_size = sum(child.params_size for child in self.children)

  @property
  def params(self):
    p = [child.params for child in self.children]
    # return np.concatenate([self._theta, *p])
    return np.concatenate(p)

  def set_params(self, x: npt.NDArray):
    start, end = 0, 0
    # start, end = 0, len(self.weights)
    # self._theta = x[start:end]
    # self.weights = softmax(x[start:end])

    for child in self.children:
      start = end
      end += child.params_size
      child.set_params(x[start:end])

  def logmar(self, query: List[Tuple[int, int]]):
    query_dict = dict(query)
    if self.i in query_dict:
      # query2 = [(i, vi) for i, vi in query if i != self.i]
      vi = query_dict.pop(self.i)

      if len(query_dict) == 0:
        return np.log(self.weights[vi])
      else:
        query2 = list(query_dict.items())
        return np.log(self.weights[vi]) + self.children[vi].logmar(query2)
    else:
      pq = [c.logmar(query) for c in self.children]
      return logsumexp(pq, b=self.weights)

  def mar_grad(self, query: List[Tuple[int, int]]):
    x = self._theta  # np.log(self.weights)
    sg = s_grad(x)
    query_dict = dict(query)
    if self.i in query_dict:
      vi = query_dict.pop(self.i)

      if len(query_dict) == 0:
        return sg[vi, :]
      else:
        query2 = list(query_dict.items())
        return sg[vi, :] * np.exp(self.children[vi].logmar(query2))
    else:
      pq = np.exp([c.logmar(query) for c in self.children])
      return np.array([np.dot(pq, sg[k, :]) for k, _ in enumerate(x)])

  def full_mar_grad(self, query: List[Tuple[int, int]]):
    # concate {grad wrt weights, grad wrt children param}
    # local_grad = self.mar_grad(query)
    g = np.zeros(self.params_size)
    # g[0:len(local_grad)] = local_grad

    query_dict = dict(query)
    if self.i in query_dict:
      vi = query_dict.pop(self.i)
      if len(query_dict) == 0:
        # query satisfied, children not accessed
        return g

      query2 = list(query_dict.items())
      start = 0  # len(local_grad)
      for k, child in enumerate(self.children):
        # vi-th child has this value, rest are 0
        if k != vi:
          start += child.params_size
        else:
          grad = self.weights[vi] * self.children[vi].full_mar_grad(query2)
          g[start:(start + len(grad))] = grad
          return g
    else:
      grads = [weight * child.full_mar_grad(query) for weight, child in zip(self.weights, self.children)]
      # g[len(local_grad):] = np.concatenate(grads)
      g = np.concatenate(grads)
      return g

  def logp(self, X: npt.NDArray):
    j = self.i
    p = np.zeros(len(X))
    for child, value, weight in zip(self.children, self.values, self.weights):
      I = X[:, j] == value
      if np.any(I):
        p[I] = child.logp(X[I]) + np.log(weight)
    return p

  def loglik(self, X: npt.NDArray):
    return np.sum(self.logp(X))
    # return sum(np.sum(X[:, self.i] == value)*np.log(weight) + child.loglik(X[X[:, self.i] == value])
    #     for value, weight, child in zip(self.values, self.weights, self.children))

  @property
  def reg(self):
    # return np.sum(log_softmax(self._theta)) + sum([child.reg for child in self.children])
    return sum([child.reg for child in self.children])

  def loglik_grad(self, X, alpha):
    x = self._theta  # np.log(self.weights)
    sg = s_grad(x)

    weights_grad = np.zeros_like(x)
    grads = []
    for k_, (child, value, weight) in enumerate(zip(self.children, self.values, self.weights)):
      I = X[:, self.i] == value
      N = np.sum(I)
      weights_grad += (N + alpha) * sg[k_, :] / weight
      grads.append(child.loglik_grad(X[I], alpha))

    # return np.concatenate([weights_grad, *grads])
    return np.concatenate(grads)

  def conditional(self, a: int, va: int, b: int, vb: int):
    # P(A | B)
    A = (a, va)
    B = (b, vb)
    return np.exp(self.logmar([A, B]) - self.logmar([B]))

  def conditional_grad(self, a: int, va: int, b: int, vb: int):
    A = (a, va)
    B = (b, vb)
    numer = np.exp(self.logmar([A, B]))
    denom = np.exp(self.logmar([B]))
    numer_grad = self.full_mar_grad([A, B])
    denom_grad = self.full_mar_grad([B])

    return numer / denom, (numer_grad * denom - denom_grad * numer) / (denom ** 2)

  def delta(self, i: int, j: int, sign: int, epsilon: float):
    # i is monotonically influenced by j, P(i|j)
    # assert self.i in (i, j)

    terms = np.array([
      np.cumsum([self.conditional(i, vi, j, vj) for vi in range(self.r[i] - 1)])
      for vj in range(self.r[j])]).T  # |j| x |i|, P(xi <= vi | xj = vj)
    rows = terms
    delta = np.array([
      sign * (row[vj2] - row[vj1]) + epsilon
      for row in rows
      for vj2, vj1 in product(range(self.r[j]), range(self.r[j]))
      if vj2 > vj1
    ])
    return delta

  def delta_grad(self, i: int, j: int, sign: int, epsilon: float):
    # i is monotonically influenced by j, P(i|j)
    # assert self.i in (i, j)

    pairs = [
      [self.conditional_grad(i, vi, j, vj) for vi in range(self.r[i] - 1)]
      for vj in range(self.r[j])]  # |j| x |i|, P(xi <= vi | xj = vj)

    delta_rows = np.array([np.cumsum([each[0] for each in row], axis=0) for row in pairs]).T
    grad_rows = np.array([np.cumsum([each[1] for each in row], axis=0) for row in pairs]).transpose(1, 0, 2)
    # terms = np.array([
    #  np.cumsum([self.conditional_grad(i, vi, j, vj) for vi in range(self.r[i]-1)], axis=0)
    #  for vj in range(self.r[j])])  # |j| x |i|, P(xi <= vi | xj = vj)

    delta = np.array([
      sign * (row[vj2] - row[vj1]) + epsilon
      for row in delta_rows
      for vj2, vj1 in product(range(self.r[j]), range(self.r[j]))
      if vj2 > vj1
    ])

    grad = np.array([
      sign * (row[vj2] - row[vj1])
      for row in grad_rows
      for vj2, vj1 in product(range(self.r[j]), range(self.r[j]))
      if vj2 > vj1
    ])
    return delta, grad

  def penalty(self, C: npt.NDArray, epsilon: float) -> float:
    scope = self.scope
    children = self.children
    pairs = [(i, j) for i, j in product(scope, scope) if i != j and C[i, j] != 0]
    penalty = 0

    for i, j in pairs:
      delta = self.delta(i, j, C[i, j], epsilon)
      penalty += np.sum((delta > 0).astype(int) * (delta ** 2))

    return penalty

  def penalty_grad(self, C: npt.NDArray, epsilon: float):
    scope = self.scope
    children = self.children
    pairs = [(i, j) for i, j in product(scope, scope) if i != j and C[i, j] != 0]
    g = np.zeros(self.params_size)
    penalty = 0
    for i, j in pairs:
      sign = C[i, j]
      delta, dg = self.delta_grad(i, j, sign, epsilon)

      penalty += np.sum((delta > 0).astype(int) * (delta ** 2))
      g += np.sum([2 * di * dgi for di, dgi in zip(delta, dg) if di > 0], axis=0)

    return penalty, g

  def fit_leaves(self, X: npt.NDArray, alpha: float, C: npt.NDArray, epsilon: float, lambda_: float, tries: int):
    for i, (child, value) in enumerate(zip(self.children, self.values)):
      X_ = X[X[:, self.i] == value]
      if isinstance(child, BaseLeaf):
        leaf = Leaf(child.scope, self.r)
        leaf.fit(X_, alpha, C, epsilon, 0)
        prev = leaf.penalty(C, epsilon)
        # .fit(X_, alpha, C, epsilon, lambda_)
        # print (lambda_, leaf.penalty(C, epsilon))
        for L in range(tries):

          if np.isclose(leaf.penalty(C, epsilon), 0):
            break

          leaf.fit(X_, alpha, C, epsilon, lambda_ * (10 ** L))
          current = leaf.penalty(C, epsilon)

          if not (current < prev):
            break
          prev = current
          # print (lambda_*(10**L), leaf.penalty(C, epsilon))

        self.children[i] = leaf
      elif isinstance(child, InternalNode):
        child.fit_leaves(X[X[:, self.i] == value], alpha, C, epsilon, lambda_, tries)
    # self.params_size = len(self.weights) + sum(child.params_size for child in self.children)
    self.params_size = sum(child.params_size for child in self.children)

  def _fit(self, X: npt.NDArray, alpha: float, C: npt.NDArray, epsilon: float, lambda_: float):
    # if np.isclose(lambda_, 0):
    #  return self
    j = self.i
    scope = self.scope
    current = np.array(self.params)

    def f(x):
      self.set_params(x)

      N = len(X)
      penalty, penalty_grad = self.penalty_grad(C, epsilon)
      """
      denom = len(X) + alpha*self.params_size + lambda_*np.count_nonzero(C)

      if N == 0:
        loss = lambda_*penalty -alpha*self.reg
        grad = lambda_*penalty_grad -self.loglik_grad(X, alpha) 
      else:
        loss = -self.loglik(X) -alpha*self.reg + lambda_*penalty 
        grad = -self.loglik_grad(X, alpha) + lambda_*penalty_grad 
      """
      denom = 1  # alpha*self.params_size + lambda_*np.count_nonzero(C) + N

      loss = (lambda_ * penalty - (self.loglik(X) + alpha * self.reg)) / denom
      # + np.linalg.norm(x-current,ord=2)**2/self.params_size
      # + 2*(x-current)/self.params_size
      grad = (lambda_ * penalty_grad - self.loglik_grad(X, alpha)) / denom
      return loss, grad

    # print (check_grad(lambda x: f(x)[0], lambda x: f(x)[1], x0=self.params, direction = "all"))
    # "L-BFGS-B"
    options = dict(maxfun=30000, maxiter=30000)
    res = minimize(f, self.params, jac=True, method="L-BFGS-B", options=options)
    if not res.success:
      print("optimization not successful")
    self.set_params(res.x)

    return self

  def fit(self, X: npt.NDArray, alpha: float, C: npt.NDArray, epsilon: float, lambda_: float, tries: int):
    # self._fit(X, alpha, C, epsilon, lambda_)
    prev = self.penalty(C, epsilon)
    for L in range(tries):
      self._fit(X, alpha, C, epsilon, lambda_ * (10 ** L))
      current = self.penalty(C, epsilon)
      if np.isclose(self.penalty(C, epsilon), 0) or (not (current < prev)):
        break

      prev = current

      # print (self.penalty(C, epsilon))

  def __repr__(self):
    return f"<InternalNode scope={', '.join([f'[{j}]' if self.i == j else f'{j}' for j in self.scope])}>"


def remove(ls: List[int], i: int):
  return [j for j in ls if j != i]


def split_data(X: npt.NDArray, r: List[int], i: int):
  splits = []
  for v in range(r[i]):
    splits.append(X[X[:, i] == v])
  return splits


# C: npt.NDArray, epsilon: float, lambda_: float,
def learn_cnet(D: Dataset, scope: List[int], alpha: float, min_instances: int = 10, min_variables: int = 5):
  # print(f"learn_cnet {len(D.X)}, {len(scope)}")
  if len(D.X) <= min_instances or len(scope) < min_variables:
    C = np.zeros((D.X.shape[1], D.X.shape[1]))
    # X: npt.NDArray, alpha: float, C: npt.NDArray, epsilon: float, lambda_: float
    return Leaf(scope, D.r).fit(D.X, alpha, C, 0, 0)

  candidates = [split_data(D.X, D.r, j) for j in scope]

  scores = [score(j, Xs, D.r, remove(scope, j), alpha) for Xs, j in zip(candidates, scope)]
  i = np.argmax(scores)
  Xs = candidates[i]
  weights = np.array([len(X) + alpha for X in Xs], dtype=float)
  weights /= np.sum(weights)

  args = (remove(scope, scope[i]), alpha, min_instances)
  children = [learn_cnet(Dataset(X, D.r), *args) for X in Xs]

  inode = InternalNode(scope[i],
                       scope,
                       D.r,
                       list(range(D.r[scope[i]])),
                       weights=weights,
                       children=children)
  return inode


def score(j: int, Xs: List[npt.NDArray], r: List[int], scope: List[int], alpha: float):
  weights = np.array([len(X) + alpha for X in Xs], dtype=float)
  weights /= np.sum(weights)
  leaves = [BaseLeaf(scope, r).fit(X, alpha) for X in Xs]
  return InternalNode(j, [j] + scope, r, list(range(r[j])), weights, leaves).loglik(np.concatenate(Xs, axis=0))


from anytree import NodeMixin, RenderTree


class WNode(NodeMixin):

  def __init__(self, foo, parent=None, weight=None):
    super(WNode, self).__init__()
    self.foo = foo
    self.parent = parent
    self.weight = weight if parent is not None else None

  def _post_detach(self, parent):
    self.weight = None



def edges(node):
  if not isinstance(node, InternalNode):
    return
  for child, value, weight in zip(node.children, node.values, node.weights):
    yield node, child, f"{value}:{weight:.4f}"
    if not isinstance(node, InternalNode): continue
    for parent, n, value in edges(child):
      yield parent, n, value


def print_cnet(node, names):
  root = WNode(names[node.i])
  nodes = {f"{names[node.i]}": root}
  # print (node)
  for parent, n, value in edges(node):
    # print (parent, n, value)
    if isinstance(n, InternalNode):
      current = WNode(f"{names[n.i]}", parent=nodes[f"{names[parent.i]}"], weight=value)
      nodes[f"{names[n.i]}"] = current
    else:
      WNode(f"Leaf({', '.join(map(lambda x: names[x], n.scope))})", parent=nodes[f"{names[parent.i]}"], weight=value)
  for pre, _, n in RenderTree(root):
    if n.weight is not None:
      print("%s%s (%s)" % (pre, n.foo, n.weight))
    else:
      print("%s%s" % (pre, n.foo))
