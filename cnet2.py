from typing import *

from itertools import product
import numpy as np
from scipy.special import logsumexp, softmax, log_softmax
from scipy.optimize import check_grad, minimize
from leaf2 import *
import numpy.typing as npt
from joblib import Parallel, delayed


class OrNode(Node):
  i: int
  r: List[int]
  values: List[int]
  weights: npt.NDArray
  children: List[ChowLiuTree]

  def __init__(self, i: int, scope: List[int], r: List[int], values: npt.NDArray,
               weights: npt.NDArray, children: List[ChowLiuTree]):
    super().__init__()
    self.scope = scope
    self.i = i  # splitting attribute
    self.r = r  
    self.values = values  # value corresponding to each child
    self.weights = weights
    self.children = children
  
  @property
  def parameter_count(self):
    return len(self.values)-1 + sum(c.parameter_count for c in self.children)
  
  
  def logmar(self, query: List[Tuple[int, int]]):
    query_dict = dict(query)
    if self.i in query_dict:
      vi = query_dict.pop(self.i) # (Vi = vi)

      if len(query_dict) == 0:
        return np.log(self.weights[vi])
      else:
        query2 = list(query_dict.items())
        return np.log(self.weights[vi]) + self.children[vi].logmar(query2)
    else:
      pq = [c.logmar(query) for c in self.children]
      return logsumexp(pq, b=self.weights)
  
  def logp(self, X: npt.NDArray):
    p = np.zeros(len(X))
    D = Dataset(X, self.r, self.scope)
    for (value, split), weight, child in zip(D.split(self.i), self.weights, self.children):
      p[X[:, self.scope.index(self.i)] == value] = child.logp(split.X) + np.log(weight)
    return p

  def loglik(self, D: Dataset):
    return sum([
      child.logp(split.X).sum() + len(split.X)*np.log(weight)
      for (value, split), weight, child in zip(D.split(self.i), self.weights, self.children)
    ])
      
   
  def conditional(self, a: int, va: int, b: int, vb: int):
    # P(A | B)
    A = (a, va)
    B = (b, vb)
    return np.exp(self.logmar([A, B]) - self.logmar([B]))

  
  def delta(self, i: int, j: int, sign: int, epsilon: float):
    # i is monotonically influenced by j, P(i|j)
    # assert self.i in (i, j)
    
    scope = self.scope
    terms = np.array([
      np.cumsum([self.conditional(i, vi, j, vj) for vi in range(self.r[scope.index(i)] - 1)])
      for vj in range(self.r[scope.index(j)])]).T  # |j| x |i|, P(xi <= vi | xj = vj)
    rows = terms
    delta = np.array([
      sign * (row[vj2] - row[vj1]) + epsilon
      for row in rows
      for vj2, vj1 in product(range(self.r[scope.index(j)]), range(self.r[scope.index(j)]))
      if vj2 > vj1
    ])
    return delta

  def penalty(self, C: npt.NDArray, epsilon: float) -> float:
    scope = self.scope
    children = self.children
    pairs = [(i, j) for i, j in product(scope, scope) if i != j and C[scope.index(i), scope.index(j)] != 0]
    penalty = 0

    for i, j in pairs:
      delta = self.delta(i, j, C[scope.index(i), scope.index(j)], epsilon)
      penalty += np.sum((delta > 0).astype(int) * (delta ** 2))

    return penalty
  
  def fit_leaves_with_knowledge(self, D: DatasetWithKnowledge, alpha: float, tries: int, scale_tries: bool = False):
    
    for index, (value, split) in enumerate(D.split(self.i)):
      if isinstance(self.children[index], BaseLeaf):
        denom = np.log10(max(10, len(split.X)))
        scaled_tries = int(np.ceil(tries / denom)) if scale_tries else tries
        leaf = Leaf(self.children[index]).fit(split, alpha, scaled_tries)
        self.children[index] = leaf
      elif isinstance(self.children[index], OrNode):
        self.children[index].fit_leaves_with_knowledge(split, alpha, tries, scale_tries)
    return self
    

  def __repr__(self):
    return f"<InternalNode scope={', '.join([f'[{j}]' if self.i == j else f'{j}' for j in self.scope])}>"


def remove(ls: List[int], i: int):
  return [j for j in ls if j != i]


def split_data(X: npt.NDArray, r: List[int], i: int):
  splits = []
  for v in range(r[i]):
    splits.append(X[X[:, i] == v])
  return splits


def learn_cnet_base(D: Dataset, alpha: float, min_instances: int = 10, min_variables: int = 5):
  """ structure learning using data """  
  # print (len(D.X), min_instances, len(D.scope), min_variables)
  if len(D.X) <= min_instances or len(D.scope) < min_variables:
    # X: npt.NDArray, alpha: float, C: npt.NDArray, epsilon: float, lambda_: float
    # print ("Leaf created")
    return BaseLeaf().fit(D, alpha)
  
  MI = compute_mutual_information(D, alpha)
  scores = np.sum(MI, axis = 0) - np.diag(MI)
  
  i = np.argmax(scores)
  # Xs = split_data(D.X, D.r, scope[i])
  values, Ds = zip(*D.split(D.scope[i]))
  
  weights = np.array([len(Di.X) + alpha for Di in Ds], dtype=float)
  weights /= np.sum(weights)
  
  # D: Dataset, scope: List[int], alpha: float, min_instances: int = 10, min_variables: int = 5
  args = (alpha, min_instances, min_variables)
  children = [learn_cnet_base(Di, *args) for Di in Ds]

  or_node = OrNode(D.scope[i],
                       D.scope,
                       D.r,
                       values,
                       weights=weights,
                       children=children)
  return or_node

def learn_cnet_with_knowledge(D: DatasetWithKnowledge, alpha: float, tries: float = 10, 
                              scale_tries: bool = False, min_instances: int = 10, min_variables: int = 5):
  """ structure learning using data and knowledge """
  denom = np.log10(max(10, len(D.X)))
  scaled_tries = int(np.ceil(tries / denom)) if scale_tries else tries
  
  # print (len(D.X), min_instances, len(D.scope), min_variables)
  if len(D.X) <= min_instances or len(D.scope) < min_variables:
    # print ("Leaf created")
    leaf = Leaf().fit(D, alpha, scaled_tries)
    return leaf
  
  """
  MI = compute_mutual_information_with_knowledge(D, alpha)
  scores = np.sum(MI, axis = 0) - np.diag(MI)
  """
  scores = compute_scores(D, alpha, scaled_tries)
  i = np.argmax(scores)
  # Xs = split_data(D.X, D.r, scope[i])
  values, Ds = zip(*D.split(D.scope[i]))
  
  weights = np.array([len(Di.X) + alpha for Di in Ds], dtype=float)
  weights /= np.sum(weights)
  
  
  # D: Dataset, alpha: float, tries: float,  
  # scale_tries: bool = True, min_instances: int = 10, min_variables: int = 5
  args = (alpha, tries, scale_tries, min_instances, min_variables)
  children = [learn_cnet_with_knowledge(Di, *args) for Di in Ds]

  or_node = OrNode(D.scope[i],
                       D.scope,
                       D.r,
                       values,
                       weights=weights,
                       children=children)
  return or_node





def compute_scores(D: DatasetWithKnowledge, alpha: float, tries: int):
  n = len(D.scope)
  scores = np.zeros(n)
  for i in range(n):
    values, Ds = zip(*D.split(D.scope[i]))

    weights = np.array([len(Di.X) + alpha for Di in Ds], dtype=float)
    weights /= np.sum(weights)

    children = [Leaf().fit(Di, alpha, tries) for Di in Ds]

    or_node = OrNode(D.scope[i],
                         D.scope,
                         D.r,
                         values,
                         weights=weights,
                         children=children)
    
    # print (scores[i])
    scores[i] = or_node.loglik(D)/len(D.X) - np.log(len(D.X))*or_node.penalty(D.C, D.epsilon)
    # print (scores[i])
    # print (scores[i])
    # scores[i] = or_node.loglik(D)/len(D.X) - np.log(len(D.X))*or_node.penalty(D.C, D.epsilon)
    # scores[i] = 2*(or_node.loglik(D)/ - len(D.X)*np.log(len(D.X))*or_node.penalty(D.C, D.epsilon)) - 2*or_node.parameter_count
    # scores[i] = 2*or_node.loglik(D) - or_node.parameter_count*np.log(len(D.X))*or_node.penalty(D.C, D.epsilon)
    # scores[i] = 2*(or_node.loglik(D) - len(D.X)*np.log(len(D.X))*or_node.penalty(D.C, D.epsilon)) #- 2*or_node.parameter_count 
  return scores

def mutual_info(i, j, clt, r):
  mi = 0
  for vi, vj in product(range(r[i]), range(r[j])):
    # Each term: P(x, y) (log P(x, y) - log P(x) - log P(y))
    t1 = clt.logmar([(i, vi), (j, vj)])
    t2 = clt.logmar([(i, vi)])
    t3 = clt.logmar([(j, vj)])

    mi += np.exp(t1) * (t1 - t2 - t3)
  return mi

def mi_score(i, leaf, r, scope):
  return sum([mutual_info(scope.index(i), scope.index(j), leaf.clt, r) for j in scope if i != j])




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
  if not isinstance(node, OrNode):
    return
  for child, value, weight in zip(node.children, node.values, node.weights):
    yield node, child, f"{value}:{weight:.4f}"
    if not isinstance(node, OrNode): continue
    for parent, n, value in edges(child):
      yield parent, n, value


def print_cnet(node, names):
  root = WNode(names[node.i])
  nodes = {f"{names[node.i]}": root}
  # print (node)
  for parent, n, value in edges(node):
    # print (parent, n, value)
    if isinstance(n, OrNode):
      current = WNode(f"{names[n.i]}", parent=nodes[f"{names[parent.i]}"], weight=value)
      nodes[f"{names[n.i]}"] = current
    else:
      WNode(f"Leaf({', '.join(map(lambda x: names[x], n.scope))})", parent=nodes[f"{names[parent.i]}"], weight=value)
  for pre, _, n in RenderTree(root):
    if n.weight is not None:
      print("%s%s (%s)" % (pre, n.foo, n.weight))
    else:
      print("%s%s" % (pre, n.foo))
