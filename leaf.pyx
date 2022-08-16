

import numpy as np
cimport numpy as np
np.import_array()

from itertools import product
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
from scipy.special import softmax, log_softmax, logsumexp
from scipy.optimize import minimize
cimport scipy.special
cimport cython

@cython.ccall
def s_grad(x: np.ndarray):
  sx = softmax(x)
  g = -np.outer(sx, sx)
  np.fill_diagonal(g, sx*(1-sx))
  return g



@cython.cclass
class Dataset:
  X = cython.declare(np.ndarray, visibility = "public")
  r = cython.declare(list, visibility = "public")

  def __init__(self, X, r):
    self.X = X
    self.r = r 

@cython.cclass
class ChowLiuTree:
  n = cython.declare(int, visibility = "public")
  parent = cython.declare(list, visibility = "public") 
  r = cython.declare(list, visibility = "public")
  log_factors = cython.declare(list, visibility = "public")
  values = cython.declare(np.ndarray, visibility = "public")
  
  def __init__(self, parent:list, r:list, log_factors:list):
    self.n = len(parent)
    self.parent = parent
    self.r = r
    self.log_factors = log_factors
    cdef int i
    self.values = np.array(list(product(*[range(r[i]) for i in range(self.n)])), dtype=int)
  
  @cython.ccall
  def logpc(self, i:int, V:np.ndarray):
    # P(xi = vi | Pa(xi) = vj); Pa(xi) = xj
    if self.parent[i] < 0:
      return self.log_factors[i][V[:, i]]
    else:
      j = self.parent[i]
      return self.log_factors[i][V[:, j], V[:, i]]
  
  @cython.ccall
  def logp(self, X:np.ndarray):
    i: int
    return np.sum([
      self.logpc(i, X)
      for i in range(self.n)
    ], axis=0)
  
  @cython.ccall
  def loglik(self, X:np.ndarray) -> double:
    return np.sum(self.logp(X))
  
  @cython.ccall
  def logmar(self, query: list) -> double:
    # [(i, vi), (j, vj), ...]
    values = self.values
    I = np.all([values[:, i] == vi for i, vi in query], axis=0)
    
    return logsumexp(self.logp(values[I]))
  
  @cython.ccall
  def conditional(self, A: tuple, B: tuple) -> double:
    # P(A|B)
    return np.exp(self.logmar([A, B]) - self.logmar([B]))
  
  @cython.ccall
  def delta(self, i:int, j:int, sign:int = 1, eps = 0.01):
    # i is monotonically influenced by j
    cdef int vi, vj, vj1, vj2
    # cdef np.ndarray[double, ndim=1] row
    terms = np.array([
      np.cumsum([self.conditional((i, vi), (j, vj)) for vi in range(self.r[i]) ])
      for vj in range(self.r[j]) 
    ]).T # |j| x |i|, P(xi <= vi | xj = vj)
    terms = terms[:-1]
    # P(xi <= vi | xj = vj2) - P(xi <= vi | xj = vj1)

    return np.array([
      sign*(row[vj2] - row[vj1]) + eps
      for row in terms
      for vj2, vj1 in product(range(self.r[j]), range(self.r[j]))
      if vj2 > vj1
    ])
  
  @cython.ccall
  def penalty(self, C: np.ndarray, eps = 0.01):
    p = 0
    for i, j in zip(*np.nonzero(C)):
      if i == j: continue

      d = self.delta(i, j, C[i, j], eps)
      p += np.sum((d > 0)*d**2)
    return p
    
     

@cython.cfunc
cdef mar_grad(clt: ChowLiuTree, factor_grads: list, query: list):
  _values = clt.values
  I = np.all([_values[:, i] == vi for i, vi in query], axis=0)
  terms = []
  cdef int a, b, c
  cdef np.ndarray[double, ndim=1] term1
  cdef np.ndarray[double, ndim=2] term2
  for a in range(clt.n):
    p = clt.parent[a]
    if p < 0:
      
      term1 = np.zeros(clt.r[a])
      for c in range(clt.r[a]):
        values = _values[I] 
        ratio = clt.logp(values) - clt.logpc(a, values)
        g = factor_grads[a] 
        term1[c] += np.sum(np.exp(ratio)*g[c][values[:, a]], axis=0)
      terms.append(np.array(term1))
    else:
      term2 = np.zeros((clt.r[p], clt.r[a]))
      #for c, b in product(range(clt.r[a]), range(clt.r[p])):
      for c in range(clt.r[a]):
        for b in range(clt.r[p]):
          values = _values[I & (clt.values[:, p]==b)]
          ratio = clt.logp(values) - clt.logpc(a, values)
          g = factor_grads[a][b]
          term2[b,c] += np.sum(np.exp(ratio)*g[c][values[:, a]], axis=0)
      terms.append(np.array(term2))
    
  return terms
  
@cython.ccall
def conditional_grad(clt: ChowLiuTree, factor_grads: list, A:tuple, B:tuple):
  # P(A|B) = P(A, B) / P(B)
  cdef double numer = np.exp(clt.logmar([A, B]))
  cdef double denom = np.exp(clt.logmar([B]))

  numer_grad = mar_grad(clt, factor_grads, [A, B])
  denom_grad = mar_grad(clt, factor_grads, [B])

  cdef int a, b, c, p
  cdef np.ndarray[double, ndim=1] term1
  cdef np.ndarray[double, ndim=2] term2
  terms = []
  for a, p in enumerate(clt.parent):
    if p < 0:  
      term1 = np.zeros(clt.r[a], dtype = float)
      for c in range(clt.r[a]):  
        term1[c] = numer_grad[a][c]*denom 
        term1[c] -= numer*denom_grad[a][c]
        term1[c] /= (denom**2)
      terms.append(np.array(term1))
    else:
      term2 = np.zeros((clt.r[p], clt.r[a]), dtype=float)
      # for b, c in product(range(clt.r[p]), range(clt.r[a])):
      for b in range(clt.r[p]):
        for c in range(clt.r[a]):

          term2[b,c] =  numer_grad[a][b,c]*denom 
          term2[b,c] -= numer*denom_grad[a][b,c]
          term2[b,c] /= (denom**2)
      terms.append(np.array(term2))
  return terms 


@cython.cfunc
def delta_grad(clt: ChowLiuTree, factor_grads: list, i:int, j:int, sign:int, eps:double):
  cg = [[conditional_grad(clt, factor_grads, (i, vi), (j, vj)) for vi in range(clt.r[i]) ]
         for vj in range(clt.r[j]) ]
  d = clt.delta(i, j, sign, eps)
  terms = [np.zeros_like(f) for f in clt.log_factors]
  cdef int a,b,c,p
  for a, p in enumerate(clt.parent):
    if p < 0:  
      for c in range(clt.r[a]):
        rows = np.array([np.cumsum([e[a][c] for e in row]) for row in cg]).T
        rows = rows[:-1]
        diffs = np.array([
          sign*(row[vj2] - row[vj1])
          for row in rows
          for vj2, vj1 in product(range(clt.r[j]), range(clt.r[j]))
          if vj2 > vj1
        ])
        terms[a][c] = np.sum(2*d*(d > 0)*diffs)
        
    else:
      # for b, c in product(range(clt.r[p]), range(clt.r[a])):
      for b in range(clt.r[p]):
        for c in range(clt.r[a]):
          rows = np.array([np.cumsum([e[a][b, c] for e in row]) for row in cg]).T
          rows = rows[:-1]
          diffs = np.array([
            sign*(row[vj2] - row[vj1])
            for row in rows
            for vj2, vj1 in product(range(clt.r[j]), range(clt.r[j]))
            if vj2 > vj1
          ])
          terms[a][b,c] = np.sum(2*d*(d > 0)*diffs)
  return terms

@cython.ccall
def penalty_grad(clt: ChowLiuTree, factor_grads:list, C:np.ndarray, eps:double):
  terms = [np.zeros_like(f) for f in clt.log_factors]
  cdef int i, j, a, b, c, p
  n = len(C)
  for i, j in zip(*np.nonzero(C)):
    dg = delta_grad(clt, factor_grads, i, j, C[i, j], eps)
    for a, p in enumerate(clt.parent):
      if p < 0:  
        for c in range(clt.r[a]):  
          terms[a][c] += dg[a][c]
      else:
        # for b, c in product(range(clt.r[p]), range(clt.r[a])):
        for b in range(clt.r[p]):
          for c in range(clt.r[a]):
            terms[a][b,c] += dg[a][b,c]
  return terms



@cython.ccall
def sufficient_stats(parent: list, D: Dataset):
  return [
    unary(i, D.X, D.r) 
      if pi < 0
      else binary(pi, i, D.X, D.r)
    for i, pi in enumerate(parent)
  ]

@cython.ccall
def pack(x: list, parent: list, r: list):
  packed = []
  for i, p in enumerate(parent):
    if p < 0:
      packed.extend(x[i]) 
    else:
      for vp in range(r[p]):
        packed.extend(x[i][vp])
  return np.array(packed)

@cython.ccall
def unpack(x: np.ndarray, parent: list, r: list):
  terms = []
  x_iter = iter(x)
  for i, p in enumerate(parent):
    if p < 0:
      terms.append(np.array([next(x_iter) for vi in range(r[i]) ]))
    else:
      terms.append(np.array([
        [next(x_iter) for vi in range(r[i]) ] 
        for vp in range(r[p])
      ]))
  return terms

@cython.ccall
def loglik_grad(clt: ChowLiuTree, ss: list):
  terms = []
  for i, pi in enumerate(clt.parent):
    if pi < 0:
      terms.append(ss[i] - np.exp(clt.log_factors[i])*np.sum(ss[i]))
    else:     
      terms.append(np.array([
        ss[i][j] - np.exp(clt.log_factors[i][j])*np.sum(ss[i][j])
        for j in range(clt.r[pi])
      ]))
  return terms


@cython.cfunc
def unary(i: int, X: np.ndarray, r: list):
  return np.array([
      np.sum((X[:, i] == v))
      for v in range(r[i])
  ])

@cython.cfunc
def binary(i:int, j:int, X: np.ndarray, r: list):
  return np.array([
      [np.sum((X[:, i] == vi)&(X[:, j] == vj)) 
        for vj in range(r[j])]
      for vi in range(r[i]) 
  ])

@cython.cfunc
def p1(i: int, X: np.ndarray, r: list, alpha: double):
  ci = unary(i, X, r) + alpha
  return ci / np.sum(ci)

@cython.cfunc
def p2(i: int, j: int, X: np.ndarray, r: list, alpha: double):
  cij = binary(i, j, X, r) + alpha
  return cij / np.sum(cij)

@cython.cfunc
def pc(i: int, j: int, X: np.ndarray, r: list, alpha: double):
  cij = binary(i, j, X, r) + alpha
  return (cij / np.sum(cij, axis = 0)).T


@cython.ccall
def fit_structure(D: Dataset, alpha: double):
  X = D.X 
  r = D.r
  n: int = X.shape[1]
  MI: np.ndarray = np.zeros((n,n))
  for i, j in product(range(n), range(n)):
    _p2: np.ndarray = p2(i, j, X, r, alpha)
    _pi1: np.ndarray = p1(i, X, r, alpha)
    _pj1: np.ndarray = p1(j, X, r, alpha)
    MI[i,j] = MI[j,i] = sum([
      _p2[vi, vj]*(np.log(_p2[vi, vj]) - np.log(_pi1[vi]) - np.log(_pj1[vj]))
        for vi, vj in product(range(r[i]), range(r[j]))
    ]) 
  MI[MI == 0.0] = 1e-6 

  mst = minimum_spanning_tree(-MI)
  dfs_tree = depth_first_order(mst, directed=False, i_start=0)
  parent = [-1 for _ in range(n)]
  for p in range(1, n):
    parent[p] = dfs_tree[1][p]
  
  return parent

@cython.ccall
def fit_base_parameters(parent: list, D: Dataset, alpha: double):
  X = D.X 
  r = D.r
  n: int = X.shape[1]
  return [
      np.log(p1(i, X, r, alpha)) 
        if parent[i] < 0 
        else np.log(pc(i, parent[i], X, r, alpha)) 
      for i in range(n)
  ]

@cython.ccall
def fit_base(D: Dataset, alpha: double):
  parent = fit_structure(D, alpha)
  log_factors = fit_base_parameters(parent, D, alpha)
  return ChowLiuTree(parent, D.r, log_factors)


@cython.ccall
def f(x: np.ndarray, parent: list, r: list, X: np.ndarray, ss: list, alpha: float, C: np.ndarray, epsilon: float, lambda_:float):
  theta = unpack(x, parent, r)
  log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1) 
                 for t, p in zip(theta, parent)]
  
  reg = sum([np.sum(e) for e in log_factors])
  tree = ChowLiuTree(parent, r, log_factors)
  if len(X) == 0:
    return - alpha*reg + lambda_*tree.penalty(C, epsilon)
  return -tree.loglik(X)/len(X) - alpha*reg/len(X) + lambda_*tree.penalty(C, epsilon)

@cython.ccall
def g(x: np.ndarray, parent: list, r: list, X: np.ndarray, ss: list, alpha: float, C: np.ndarray, epsilon: float, lambda_:float):
  theta = unpack(x, parent, r)

  log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1) 
                    for t, p in zip(theta, parent)]
  factor_grads = [
      s_grad(t) if p  < 0 else np.array([s_grad(t[vp]) for vp in range(r[p]) ])
      for t, p in zip(theta, parent)
  ]

  reg_grad = [ 
      np.sum(factor_grads[i]/np.exp(log_factors[i]), axis = 1)
      if pi < 0
      else [np.sum(factor_grads[i][j]/np.exp(log_factors[i][j]), axis = 1) for j in range(r[pi])]
      for i, pi in enumerate(parent)
  ]
  tree = ChowLiuTree(parent, r, log_factors)
  p_terms = penalty_grad(tree, factor_grads, C, epsilon)
  if len(X) == 0:
    return - alpha*pack(reg_grad, parent, r) + lambda_*pack(p_terms, parent, r)
  
  ll_terms = loglik_grad(tree, ss)
  return -pack(ll_terms, parent, r)/len(X) - alpha*pack(reg_grad, parent, r)/len(X) + lambda_*pack(p_terms, parent, r)


@cython.ccall
def fit_grad(D: Dataset, C: np.ndarray,  epsilon: float, lambda_:float, alpha: double, parent: list = None):
  if parent is None:
    parent = fit_structure(D, alpha)
  ss = sufficient_stats(parent, D)
  log_factors = fit_base_parameters(parent, D, alpha)
  if lambda_ == 0:
    return ChowLiuTree(parent, D.r, log_factors)
  init = pack(log_factors, parent, D.r)
  args = (parent, D.r, D.X, ss, alpha, C, epsilon, lambda_)
  res = minimize(f, init, jac=g, args = args)

  theta = unpack(res.x, parent, D.r)
  log_factors = [log_softmax(t) if p < 0 else log_softmax(t, axis=1) 
                for t, p in zip(theta, parent)]

  return ChowLiuTree(parent, D.r, log_factors)

