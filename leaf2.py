from typing import *
import numpy.typing as npt

from itertools import islice
from itertools import product
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize, check_grad
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import softmax, log_softmax, logsumexp
from scipy.special import xlogy

seeds = [29128, 70796, 35117, 72774, 59670, 18922, 28321, 59607, 38085, 34675]

def s_grad(x: np.ndarray):
  sx = softmax(x)
  g = -np.outer(sx, sx)
  np.fill_diagonal(g, sx * (1 - sx))
  return g


class Dataset:
  X: np.ndarray
  r: list
  scope: list
  
  def __init__(self, X: np.ndarray, r: list, scope = None):
    self.X = X
    self.r = r
    if scope is None:
      self.scope = list(range(len(r))) # scope
    else:
      self.scope = list(scope)
    
  def split(self, v):
    remaining = [ri for ri, vi in zip(self.r, self.scope) if vi != v]
    scope = [vi for vi in self.scope if vi != v]
    index = [i for i, vi in enumerate(self.scope) if vi != v]
    Xv = self.X[:, self.scope.index(v)]
    return [
        (x, Dataset(
          self.X[Xv == x][:, index], 
          remaining, 
          scope
        ))
        for x in range(self.r[self.scope.index(v)])
    ]
  


class DatasetWithKnowledge(Dataset):
  X: np.ndarray
  r: list
  scope: list
  C: np.ndarray
  epsilon: float
  
  def __init__(self, X: np.ndarray, r: list, C: np.ndarray, epsilon: float, scope = None):
    super().__init__(X, r, scope)
    self.C = C.astype(int)
    self.epsilon = epsilon
  
  def add_noise(self, noise: float = 0.3):
    # Add noise
    # for every X inf+ Y, replace noise % of Y with r[Y] - ceil(X*r[Y]/r[X])
    gen = np.random.default_rng(seed = seeds[0])
    X = np.array(self.X)
    noise_size = int(len(X) * noise)
    for i, j in zip(*np.nonzero(self.C)):
      ratio = (self.r[i] - 1)/(self.r[j] - 1)
      if self.C[i, j] == +1:
        gen.shuffle(X)
        X[:noise_size, i] = self.r[i] - 1 - np.floor(self.X[:noise_size, j]*ratio )
      else:
        gen.shuffle(X)
        X[:noise_size, i] = np.floor(self.X[:noise_size, j]*ratio)
    gen.shuffle(X)
    return DatasetWithKnowledge(
          X, 
          self.r,
          self.C,
          self.epsilon,
          self.scope
    )
  
  def split(self, v):
    remaining = [ri for ri, vi in zip(self.r, self.scope) if vi != v]
    scope = [vi for vi in self.scope if vi != v]
    index = [i for i, vi in enumerate(self.scope) if vi != v]
    Xv = self.X[:, self.scope.index(v)]

    return [
        (x, DatasetWithKnowledge(
          self.X[Xv == x][:, index], 
          remaining,
          self.C[index, :][:, index],
          self.epsilon,
          scope
        ))
        for x in range(self.r[self.scope.index(v)])
    ]
  
  def subset(self, variables):
    remaining = [self.r[self.scope.index(vi)] for vi in variables]
    scope = variables
    index = [self.scope.index(vi) for vi in variables]
    
    return DatasetWithKnowledge(
          self.X[:, index], 
          remaining,
          self.C[index, :][:, index],
          self.epsilon,
          scope
    )
  
  def bootstrap_samples(self, k: int = 5):
    samples = []
    for seed in seeds[:k]:
      gen = np.random.default_rng(seed = seed)
      N = len(self.X)
      i = gen.choice(np.arange(N), replace=True, size=N)
      X = np.array(self.X[i])
      
      samples.append(DatasetWithKnowledge(
            X, 
            self.r,
            self.C,
            self.epsilon,
            self.scope
      ))
    return samples
  

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
    self.params_size = sum(e.size for e in log_factors)
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
  return np.array(packed, dtype = float)


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


def compute_mutual_information(D: Dataset, alpha: float):
  X = D.X
  r = D.r
  n: int = len(D.scope)
  MI: np.ndarray = np.zeros((n, n))
  for i, j in combinations_with_replacement(range(n), r=2):
    if i == j:
      _p = p1(i, X, r, alpha)
      MI[i, i] = -np.sum(xlogy(_p, _p))
    else:
      _p2 = p2(i, j, X, r, alpha)
      _pi1 = _p2.sum(axis = 1)
      _pj1 = _p2.sum(axis = 0)
      MI[i, j] = MI[j, i] = sum([
        _p2[vi, vj] * (np.log(_p2[vi, vj]) - np.log(_pi1[vi]) - np.log(_pj1[vj]))
        for vi, vj in product(range(r[i]), range(r[j]))
      ]) 
  return np.clip(MI, 1e-20, None)
  
def compute_mutual_information_with_knowledge2(D: DatasetWithKnowledge, alpha: float, tries: int = 10):
  X = D.X
  r = D.r
  n: int = len(D.scope)
  MI: np.ndarray = np.zeros((n, n))
  for i, j in combinations_with_replacement(range(n), r=2):
    if i == j:
      _p = p1(i, X, r, alpha)
      MI[i, i] = -np.sum(xlogy(_p, _p))
      continue
      
    if not np.all(D.C[(i,j), :][:, (i,j)] == 0):
      
      D_ = D.subset([D.scope[i], D.scope[j]])
      assert np.all(D.C[(i,j), :][:, (i,j)] == D_.C)
      parent = [-1, 0]
      log_factors = fit_base_parameters(parent, D_, alpha)
      clt = ChowLiuTree(parent, D_.r, log_factors)
      prev = clt.penalty(D_.C, D_.epsilon)

      for L in range(tries):
        if np.isclose(prev, 0):
          break

        # parent: list, D: Dataset, alpha: float, lambda_: float
        clt.log_factors = fit_grad_parameters(parent, D_, alpha, (10 ** L))

        current = clt.penalty(D_.C, D_.epsilon)
        if not (current < prev):
          break
        prev = current
      
      _p2 = np.zeros((r[i], r[j]))
      for v in np.ndindex(r[i], r[j]):
        _p2[v] = np.exp(clt.logp(np.array(v)[None, :]))[0]
    else:
      _p2 = p2(i, j, X, r, alpha)
    _pi1 = _p2.sum(axis = 1)
    _pj1 = _p2.sum(axis = 0)
    MI[i, j] = MI[j, i] = sum([
      _p2[vi, vj] * (np.log(_p2[vi, vj]) - np.log(_pi1[vi]) - np.log(_pj1[vj]))
      for vi, vj in product(range(r[i]), range(r[j]))
    ]) 
  return np.clip(MI, 1e-20, None)

def compute_mutual_information_with_knowledge(D: DatasetWithKnowledge, alpha: float, tries: int = 10):
  # fit chow-liu tree
  parent = fit_structure(D, alpha)
  log_factors = fit_base_parameters(parent, D, alpha)
  clt = ChowLiuTree(parent, D.r, log_factors)
  prev = clt.penalty(D.C, D.epsilon)

  for L in range(tries):
    if np.isclose(prev, 0):
      break
    
    clt.log_factors = fit_grad_parameters(parent, D, alpha, (10 ** L))

    current = clt.penalty(D.C, D.epsilon)
    if not (current < prev):
      break
    prev = current
  
  # estimate MI from tree
  n = len(D.scope)
  C = D.C
  r = D.r
  MI = np.zeros((n, n))
  for i, j in combinations_with_replacement(range(n), r=2):
    if i == j:
      _p = np.zeros(r[i])
      for v in range(r[i]):
        _p[v] = np.exp(clt.logmar([(i, v)]))
      MI[i, i] = -np.sum(xlogy(_p, _p))
    else:    
      _p2 = np.zeros((r[i], r[j]))
      for vi, vj in np.ndindex(r[i], r[j]):
        _p2[vi, vj] = np.exp(clt.logmar([(i, vi), (j, vj)]))
      
      
      _pi1 = _p2.sum(axis = 1)
      _pj1 = _p2.sum(axis = 0)
      
      MI[i, j] = MI[j, i] = sum([
        _p2[vi, vj] * (np.log(_p2[vi, vj]) - np.log(_pi1[vi]) - np.log(_pj1[vj]))
        for vi, vj in np.ndindex(r[i], r[j])
      ]) 
  return np.clip(MI, 1e-20, None)

def fit_structure(D: Dataset, alpha: float):
  X = D.X
  r = D.r
  n: int = X.shape[1]
  MI: np.ndarray = compute_mutual_information(D, alpha)
  mst = minimum_spanning_tree(-MI)
  dfs_tree = depth_first_order(mst, directed=False, i_start=0)
  parent = [-1 for _ in range(n)]
  for p in range(1, n):
    parent[p] = dfs_tree[1][p]

  return parent

def fit_structure_with_knowledge(D: DatasetWithKnowledge, alpha: float):
  n: int = D.X.shape[1]
  MI: np.ndarray = compute_mutual_information_with_knowledge(D, alpha)
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


def f(x: np.ndarray, parent: list, r: list, X: np.ndarray, ss: list, alpha: float, C: np.ndarray, epsilon: float,
      lambda_: float):
  theta = unpack(x, parent, r)
  log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1)
                 for t, p in zip(theta, parent)]

  reg = sum([np.sum(e) for e in log_factors])
  tree = ChowLiuTree(parent, r, log_factors)
  denom = len(X) + alpha * len(x) + lambda_ * np.count_nonzero(C)
  if len(X) == 0:
    return (-alpha * reg + lambda_ * tree.penalty(C, epsilon)) / denom
  return (-tree.loglik(X) - alpha * reg + lambda_ * tree.penalty(C, epsilon)) / denom


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
    np.sum(factor_grads[i] * np.exp(-log_factors[i]), axis=1)
    if pi < 0
    else [np.sum(factor_grads[i][j] * np.exp(-log_factors[i][j]), axis=1) for j in range(r[pi])]
    for i, pi in enumerate(parent)
  ]
  tree = ChowLiuTree(parent, r, log_factors)
  p_terms = penalty_grad(tree, factor_grads, C, epsilon)
  denom = len(X) + alpha * len(x) + lambda_ * np.count_nonzero(C)
  if len(X) == 0:
    return (-alpha * pack(reg_grad, parent, r) + lambda_ * pack(p_terms, parent, r)) / denom

  ll_terms = loglik_grad(tree, ss)
  return (-pack(ll_terms, parent, r) - alpha * pack(reg_grad, parent, r) + lambda_ * pack(p_terms, parent, r)) / denom

def fit_grad_parameters(parent: list, D: DatasetWithKnowledge, alpha: float, lambda_: float, init_log_factors = None):
  ss = sufficient_stats(parent, D)
  if init_log_factors is None:
    log_factors = fit_base_parameters(parent, D, alpha)
  else:
    log_factors = init_log_factors
  base = ChowLiuTree(parent, D.r, log_factors)
  
  if lambda_ == 0:
    return base
  if np.isclose(base.penalty(D.C, D.epsilon), 0):
    # print (np.isclose(base.penalty(C, epsilon), 0))
    return base

  init = pack(log_factors, parent, D.r)
  args = (parent, D.r, D.X, ss, alpha, D.C, D.epsilon, lambda_)
  # print (check_grad(f, g, init, *args))
  # print (g(init, *args))
  options = dict(maxfun=30000, maxiter=30000)
  res = minimize(f, init, jac=g, args=args, method="L-BFGS-B", options=options)

  theta = unpack(res.x, parent, D.r)
  log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1)
                 for t, p in zip(theta, parent)]

  return log_factors


class Node:
  scope: List[int]
  
  def __init__(self):
    self.scope = []
    
  def __repr__(self):
    return f"<{self.__class__.__name__} scope={self.scope}>"


class BaseLeaf(Node):
  scope: List[int]
  r: List[int]
  clt: ChowLiuTree

  def __init__(self):
    super().__init__()
    self.r = []
  
  @property
  def parameter_count(self):
    return sum([len(f)-1 if p < 0 else f.shape[0]*(f.shape[1]-1) for p, f in zip(self.clt.parent, self.clt.log_factors)])
    
  def fit(self, D: Dataset, alpha: float):
    self.r = D.r
    self.scope = D.scope
    parent = fit_structure(D, alpha)
    log_factors = fit_base_parameters(parent, D, alpha)
    self.clt = ChowLiuTree(parent, D.r, log_factors)
    return self

  def loglik(self, D: Dataset):
    assert D.scope == self.scope
    assert D.r == self.r
    
    return self.clt.loglik(D.X)

  def logmar(self, query: List[Tuple[int, int]]):
    q = [(self.scope.index(i), vi) for (i, vi) in query]
    return self.clt.logmar(q)

  def logp(self, X: npt.NDArray):
    assert X.shape[1] == len(self.scope)
    
    return self.clt.logp(X)

  def penalty(self, C: npt.NDArray, epsilon: float) -> float:
    assert C.shape[0] == C.shape[1]
    assert C.shape[0] == len(self.scope)
    
    return self.clt.penalty(C, epsilon)
  
  def delta(self, i: int, j: int, sign: int, epsilon: float) -> float:
    return self.clt.delta(self.scope.index(i), self.scope.index(j), sign, epsilon)

class Leaf(BaseLeaf):
  scope: List[int]
  r: List[int]
  clt: ChowLiuTree

  def __init__(self, leaf = None):
    super().__init__()
    if leaf is not None:
      self.scope = leaf.scope
      self.r = leaf.r
      self.clt = leaf.clt
  
  def fit(self, D: DatasetWithKnowledge, alpha: float, tries: int):
    self.r = D.r
    self.scope = D.scope
    
    if not hasattr(self, "clt"): 
      parent = fit_structure(D, alpha)
      log_factors = fit_base_parameters(parent, D, alpha)
      clt = ChowLiuTree(parent, D.r, log_factors)
    else:
      clt = self.clt
      parent = clt.parent
      log_factors = clt.log_factors
      
    prev = clt.penalty(D.C, D.epsilon)
    
    for L in range(tries):
      if np.isclose(prev, 0):
        break
      
      # parent: list, D: Dataset, alpha: float, lambda_: float
      clt.log_factors = fit_grad_parameters(parent, D, alpha, (10 ** L), clt.log_factors)

      current = clt.penalty(D.C, D.epsilon)
      if not (current < prev):
        break
      prev = current
    
    self.clt = clt
    
    return self

def format_influences(C, names):
  return [
    f"{names[j]} ≺ᴹ⁺ {names[i]}"
    if C[i, j] == +1
    else f"{names[j]} ≺ᴹ⁻ {names[i]}"
    for i, j in zip(*np.nonzero(C))
  ]