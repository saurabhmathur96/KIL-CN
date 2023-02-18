import numpy as np
import pandas as pd

from os import path
from itertools import product
from joblib import load

from sklearn.metrics import mean_squared_error
from pgmpy.inference import VariableElimination
from dataset import benchmark_datasets, healthcare_datasets, bns

model_dir = "models"
processed_data_dir = "Processed data"


def query1(net, C, r):
  predictions = []
  for i, j in zip(*np.nonzero(C)):
    first, second = names[i], names[j]
    
    for vj, vi in product(range(r[j]), range(r[i])):
      logp = net.logmar([(i, vi), (j, vj)]) - net.logmar([(j, vj)])
      conditional = np.exp(logp)
      predictions.append(conditional)

  return np.array(predictions)

# datasets = bns + healthcare_datasets
rows = []

for name in bns:
  dataset_path = path.join(processed_data_dir, f"{name}.joblib")
  train, test, names, bn = load(dataset_path)
    
  inference = VariableElimination(bn)
  gt = []

  for i, j in zip(*np.nonzero(train.C)):
    first, second = names[i], names[j]
    
    numerator = inference.query([first, second], show_progress=False)
    denominator = inference.query([second], show_progress=False)

    for vj, vi in product(range(train.r[j]), range(train.r[i])):
      conditional = numerator.get_value(**{first: vi, second: vj})/denominator.get_value(**{second: vj})
      gt.append(conditional)
  
  gt = np.array(gt)
  
  model_path = path.join(model_dir, f"{name}.joblib")
  cnets = load(model_path)
  errors = [[mean_squared_error(gt, query1(node, train.C, train.r)) for node in row] for row in cnets]
  
  # print (np.mean(results, axis=0).round(5))
  print ([name] + np.mean(errors, axis = 0).tolist())
  rows.append([name] + np.mean(errors, axis = 0).tolist())

def query2(cnet, target, X_test):
  y_pred = []
  # y_true = []
  for row in X_test:
    # P(Y = 1 | X = x)
    n_query = [(i, vi) if i != target else (i, 1) for i, vi in enumerate(row)]
    d_query = [(i, vi) for i, vi in enumerate(row) if i != target]

    y_pred.append(np.exp(cnet.logmar(n_query) - cnet.logmar(d_query)))
    
  return np.array(y_pred) # mean_squared_error(y_true, y_pred)


for name in benchmark_datasets + healthcare_datasets:
  dataset_path = path.join(processed_data_dir, f"{name}.joblib")
  train, test, names = load(dataset_path)
  
  target = { 
    "auto": 0,
    "ljubljana": 0,
    "numom2b_mixed_a": 2,
    "numom2b_mixed_b": 3,
    "adni": 6,
    "ppd": 10,
    "cleveland": 11
  }
  target_i = target.get(name, len(names)-1)
  gt = np.array(test.X[:, target_i])
  try:
    model_path = path.join(model_dir, f"{name}.joblib")
    cnets = load(model_path)
  except FileNotFoundError:
    continue
  
  errors = [[mean_squared_error(gt, query2(node, target_i, test.X)) for node in row] for row in cnets]

  print ([name] + np.mean(errors, axis = 0).tolist())
  rows.append([name] + np.mean(errors, axis = 0).tolist())


columns = ["Data set", "LearnCNet", "KIL-CN(-S)", "KIL-CN"]
df = pd.DataFrame(rows, columns = columns)
# print (df)
print (df.to_latex(index = False, float_format = "{0:,.4f}".format))
