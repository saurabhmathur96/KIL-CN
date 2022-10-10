from models import *
from dataset import get_bn_dataset, format_influences, bns
from pprint import pprint
from joblib import Parallel, delayed
import numpy as np
from itertools import chain
import pickle
import sys

repeat = 5
alpha = 1
epsilon = 0.001

start, end = int(sys.argv[1]), int(sys.argv[2])

for name in bns[start:end]:
  X_train, X_test, r, C, names, bn = get_bn_dataset(name, N=100)
  n = len(r)

  N = len(X_train)
  n = X_train.shape[1]

  np.random.seed(0)


  def fit_models(size):
    i = np.random.choice(np.arange(N), replace=True, size=size)

    base = CNetEstimator(r, alpha)
    local_soft = CNetLocalSoftEstimator(C, r, alpha, epsilon)
    global_soft = CNetGlobalSoftEstimator(C, r, alpha, epsilon)
    local_hard = CNetLocalHardEstimator(C, r, alpha, epsilon)
    global_hard = CNetGlobalHardEstimator(C, r, alpha, epsilon)
    models = [base, local_soft, global_soft, local_hard, global_hard]

    return [m.fit(X_train[i]) for m in models]


  result = Parallel(n_jobs=-1, verbose=10)([delayed(fit_models)(N) for _ in range(repeat)])
  print(result)
  with open(f"{name}.joblib", 'wb') as pfile:
    pickle.dump(result, pfile, protocol=pickle.HIGHEST_PROTOCOL)
