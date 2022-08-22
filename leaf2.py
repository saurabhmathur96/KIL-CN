from itertools import islice
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import softmax, log_softmax, logsumexp


def s_grad(x: np.ndarray):
  sx = softmax(x)
  g = -np.outer(sx, sx)
  np.fill_diagonal(g, sx * (1 - sx))
  return g


class Dataset:
  X: np.ndarray
  r: list

  def __init__(self, X, r):
    self.X = X
    self.r = r


class ChowLiuTree:
  n: int
  parent: list
  r: list
  log_factors: list
  values: np.ndarray

  def __init__(self, parent: list, r: list, log_factors: list):
    self.n = len(parent)
    self.parent = parent
    self.r = r
    self.log_factors = log_factors
    ranges = [np.arange(ri) for ri in r]
    self.values = np.array(np.meshgrid(*ranges)).T.reshape(-1, self.n)

  def logpc(self, i: int, V: np.ndarray):
    # P(xi = vi | Pa(xi) = vj); Pa(xi) = xj
    if self.parent[i] < 0:
      return self.log_factors[i][V[:, i]]
    else:
      j = self.parent[i]
      return self.log_factors[i][V[:, j], V[:, i]]

  def logp(self, X: np.ndarray) -> np.ndarray:
    i: int
    return np.sum([
      self.logpc(i, X)
      for i in range(self.n)
    ], axis=0)

  def loglik(self, X: np.ndarray):
    return np.sum(self.logp(X))

  def logmar(self, query: list) -> float:
    # [(i, vi), (j, vj), ...]

    values = self.values
    I = np.all([values[:, i] == vi for i, vi in query], axis=0)

    return logsumexp(self.logp(values[I]))

  def conditional(self, A: tuple, B: tuple):
    # P(A|B)
    return np.exp(self.logmar([A, B]) - self.logmar([B]))

  def delta(self, i: int, j: int, sign: int = 1, eps=0.01):
    # i is monotonically influenced by j
    # cdef np.ndarray[double, ndim=1] row
    terms = np.array([
      np.cumsum([self.conditional((i, vi), (j, vj)) for vi in range(self.r[i] - 1)])
      for vj in range(self.r[j])
    ]).T  # |j| x |i|, P(xi <= vi | xj = vj)
    # P(xi <= vi | xj = vj2) - P(xi <= vi | xj = vj1)

    return np.array([
      sign * (row[vj2] - row[vj1]) + eps
      for row in terms
      for vj2, vj1 in product(range(self.r[j]), range(self.r[j]))
      if vj2 > vj1
    ])

  def penalty(self, C: np.ndarray, eps=0.01):
    p = 0
    for i, j in zip(*np.nonzero(C)):
      if i == j: continue

      d = self.delta(i, j, C[i, j], eps)
      p += np.sum((d > 0) * d ** 2)
    return p


def mar_grad(clt, factor_grads, query):
  _values = clt.values
  _logp = clt.logp(_values)

  I = np.all([_values[:, i] == vi for i, vi in query], axis=0)
  terms = [np.zeros_like(f) for f in clt.log_factors]
  for a in range(clt.n):
    p = clt.parent[a]
    if p < 0:
      values = _values[I]
      ratio = np.exp(_logp[I] - clt.logpc(a, values))[:, None]
      g = factor_grads[a]
      terms[a] += np.sum(ratio * g[values[:, a]], axis=0)
    else:
      for b in range(clt.r[p]):
        values = _values[I & (clt.values[:, p] == b)]
        ratio = np.exp(_logp[I & (clt.values[:, p] == b)] - clt.logpc(a, values))[:, None]
        g = factor_grads[a][b]
        terms[a][b] += np.sum(ratio * g[values[:, a]], axis=0)
  return terms


def conditional_grad(clt, factor_grads, A, B):
  # P(A|B) = P(A, B) / P(B)
  numer = np.exp(clt.logmar([A, B]))
  denom = np.exp(clt.logmar([B]))

  numer_grad = mar_grad(clt, factor_grads, [A, B])
  denom_grad = mar_grad(clt, factor_grads, [B])

  terms = [np.zeros_like(f) for f in clt.log_factors]
  for a, (n_grad, d_grad) in enumerate(zip(numer_grad, denom_grad)):
    # for c in range(clt.r[a]):
    terms[a] = (n_grad * denom \
                - numer * d_grad) \
               / (denom ** 2)

  return terms


def delta_grad(clt, factor_grads, i, j, sign, eps):
  cg = [[conditional_grad(clt, factor_grads, (i, vi), (j, vj)) for vi in range(clt.r[i] - 1)]
        for vj in range(clt.r[j])]
  d = clt.delta(i, j, sign, eps)
  terms = [np.zeros_like(f) for f in clt.log_factors]
  for a, p in enumerate(clt.parent):
    if p < 0:
      for c in range(clt.r[a]):
        rows = np.array([np.cumsum([e[a][c] for e in row]) for row in cg]).T

        diffs = np.fromiter((
          sign * (row[vj2] - row[vj1])
          for row in rows  # [:-1]
          for vj2, vj1 in product(range(clt.r[j]), range(clt.r[j]))
          if vj2 > vj1
        ), dtype=float)
        terms[a][c] = np.sum(2 * d * (d > 0) * diffs)

    else:
      for b in range(clt.r[p]):
        for c in range(clt.r[a]):
          rows = np.array([np.cumsum([e[a][b, c] for e in row]) for row in cg]).T

          diffs = np.fromiter((
            sign * (row[vj2] - row[vj1])
            for row in rows
            for vj2, vj1 in product(range(clt.r[j]), range(clt.r[j]))
            if vj2 > vj1
          ), dtype=float)
          terms[a][b, c] = np.sum(2 * d * (d > 0) * diffs)
  return terms


def penalty_grad(clt, factor_grads, C, eps):
  terms = [np.zeros_like(f) for f in clt.log_factors]
  n = len(C)
  for i, j in zip(*np.nonzero(C)):
    dg = delta_grad(clt, factor_grads, i, j, C[i, j], eps)
    for a, term in enumerate(dg):
      terms[a] += term
  return terms


def sufficient_stats(parent: list, D: Dataset):
  return [
    unary(i, D.X, D.r)
    if pi < 0
    else binary(pi, i, D.X, D.r)
    for i, pi in enumerate(parent)
  ]


def pack(x: list, parent: list, r: list):
  packed = []
  for i, p in enumerate(parent):
    if p < 0:
      packed.extend(x[i])
    else:
      for vp in range(r[p]):
        packed.extend(x[i][vp])
  return np.array(packed)





def unpack(x: np.ndarray, parent: list, r: list):
  x_iter = iter(x)
  return [
    np.fromiter(islice(x_iter, r[i]), dtype=np.float)
    if p < 0
    else np.fromiter(islice(x_iter, r[i] * r[p]), dtype=np.float).reshape((r[p], r[i]))
    for i, p in enumerate(parent)
  ]


def loglik_grad(clt: ChowLiuTree, ss: list):
  return [
    ssi - np.exp(lfi) * np.sum(ssi)
    if p < 0
    else ssi - np.exp(lfi) * np.sum(ssi, axis=1)[:, None]
    for ssi, lfi, p in zip(ss, clt.log_factors, clt.parent)
  ]


def unary(i: int, X: np.ndarray, r: list):
  return np.array([
    np.sum((X[:, i] == v))
    for v in range(r[i])
  ])


def binary(i: int, j: int, X: np.ndarray, r: list):
  return np.array([
    [np.sum((X[:, i] == vi) & (X[:, j] == vj))
     for vj in range(r[j])]
    for vi in range(r[i])
  ])


def p1(i: int, X: np.ndarray, r: list, alpha: float):
  ci = unary(i, X, r) + alpha
  return ci / np.sum(ci)


def p2(i: int, j: int, X: np.ndarray, r: list, alpha: float):
  cij = binary(i, j, X, r) + alpha
  return cij / np.sum(cij)


def pc(i: int, j: int, X: np.ndarray, r: list, alpha: float):
  cij = binary(i, j, X, r) + alpha
  return (cij / np.sum(cij, axis=0)).T


def fit_structure(D: Dataset, alpha: float):
  X = D.X
  r = D.r
  n: int = X.shape[1]
  MI: np.ndarray = np.zeros((n, n))
  for i, j in product(range(n), range(n)):
    _p2: np.ndarray = p2(i, j, X, r, alpha)
    _pi1: np.ndarray = p1(i, X, r, alpha)
    _pj1: np.ndarray = p1(j, X, r, alpha)
    MI[i, j] = MI[j, i] = sum([
      _p2[vi, vj] * (np.log(_p2[vi, vj]) - np.log(_pi1[vi]) - np.log(_pj1[vj]))
      for vi, vj in product(range(r[i]), range(r[j]))
    ])
  MI[np.isclose(MI, 0)] = -1e-6
  mst = minimum_spanning_tree(-MI)
  dfs_tree = depth_first_order(mst, directed=False, i_start=0)
  parent = [-1 for _ in range(n)]
  for p in range(1, n):
    parent[p] = dfs_tree[1][p]

  return parent


def fit_base_parameters(parent: list, D: Dataset, alpha: float):
  X = D.X
  r = D.r
  n: int = X.shape[1]
  return [
    np.log(p1(i, X, r, alpha))
    if parent[i] < 0
    else np.log(pc(i, parent[i], X, r, alpha))
    for i in range(n)
  ]


def fit_base(D: Dataset, alpha: float):
  parent = fit_structure(D, alpha)
  log_factors = fit_base_parameters(parent, D, alpha)
  return ChowLiuTree(parent, D.r, log_factors)


def f(x: np.ndarray, parent: list, r: list, X: np.ndarray, ss: list, alpha: float, C: np.ndarray, epsilon: float,
      lambda_: float):
  theta = unpack(x, parent, r)
  log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1)
                 for t, p in zip(theta, parent)]

  reg = sum([np.sum(e) for e in log_factors])
  tree = ChowLiuTree(parent, r, log_factors)
  if len(X) == 0:
    return -alpha * reg + lambda_ * tree.penalty(C, epsilon)
  return -tree.loglik(X) / len(X) - alpha * reg / len(X) + lambda_ * tree.penalty(C, epsilon)


def g(x: np.ndarray, parent: list, r: list, X: np.ndarray, ss: list, alpha: float, C: np.ndarray, epsilon: float,
      lambda_: float):
  theta = unpack(x, parent, r)

  log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1)
                 for t, p in zip(theta, parent)]
  factor_grads = [
    s_grad(t) if p < 0 else np.apply_along_axis(s_grad, 1, t)  # np.array([s_grad(t[vp]) for vp in range(r[p]) ])
    for t, p in zip(theta, parent)
  ]

  reg_grad = [
    np.sum(factor_grads[i] / np.exp(log_factors[i]), axis=1)
    if pi < 0
    else [np.sum(factor_grads[i][j] / np.exp(log_factors[i][j]), axis=1) for j in range(r[pi])]
    for i, pi in enumerate(parent)
  ]
  tree = ChowLiuTree(parent, r, log_factors)
  p_terms = penalty_grad(tree, factor_grads, C, epsilon)
  if len(X) == 0:
    return -alpha * pack(reg_grad, parent, r) + lambda_ * pack(p_terms, parent, r)

  ll_terms = loglik_grad(tree, ss)
  return -pack(ll_terms, parent, r) / len(X) - alpha * pack(reg_grad, parent, r) / len(X) + lambda_ * pack(p_terms,
                                                                                                           parent, r)


def fit_grad(D: Dataset, C: np.ndarray, epsilon: float, lambda_: float, alpha: float, parent: list = None):
  if parent is None:
    parent = fit_structure(D, alpha)
  ss = sufficient_stats(parent, D)
  log_factors = fit_base_parameters(parent, D, alpha)
  if lambda_ == 0:
    return ChowLiuTree(parent, D.r, log_factors)
  init = pack(log_factors, parent, D.r)
  args = (parent, D.r, D.X, ss, alpha, C, epsilon, lambda_)
  res = minimize(f, init, jac=g, args=args, method="L-BFGS-B")

  theta = unpack(res.x, parent, D.r)
  log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1)
                 for t, p in zip(theta, parent)]

  return ChowLiuTree(parent, D.r, log_factors)
