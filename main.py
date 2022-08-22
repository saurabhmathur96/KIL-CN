# import pyximport; pyximport.install(language_level = 3)
from leaf2 import fit_base, Dataset, fit_base, fit_grad
import pandas as pd
import numpy as np

np.random.seed(0)
N, n, ri = 2000, 5, 4
X = np.random.uniform(0, ri, size = (N, n)).astype(int)
r = [ ri for i in range(n) ]
alpha = 1
clt = fit_base(Dataset(X, r), alpha)

C = np.zeros((n,n))
C[0, 1] = 1
C[0, 4] = 1
C[2, 3] = 1
eps = 0.001

D = Dataset(X, r)
clt = fit_base(D, alpha)
print (f"{clt.loglik(X):.4f}, {clt.penalty(C, eps):.4f}")

epsilon, lambda_ = 0.001, 10
clt2 = fit_grad(D, C,  epsilon, lambda_, alpha)
print (f"{clt2.loglik(X):.4f}, {clt2.penalty(C, eps):.4f}")

def gen_data(N, n, r, C):
  X = np.random.uniform(0, 1, size=(N,n))
  for i in range(n):
    if np.count_nonzero(C[i]) == 0: continue
    
  # for i, j in zip(*np.nonzero(C)):
    X[:, i] = sum([2*C[i, j]*X[:, j]  for j in np.flatnonzero(C[i])]) 

  for i in range(n):
    values = pd.cut(X[:, i], r, labels = range(r))
    X[:, i] = values.to_numpy().astype(int)
  return X.astype(int)



Xt = gen_data(N, n, ri, C)
print (f"{clt.loglik(Xt):.4f}, {clt2.loglik(Xt):.4f}")