import numpy as np
import pandas as pd

from os import path
from joblib import load
from cnet2 import edges, BaseLeaf
from dataset import benchmark_datasets, healthcare_datasets, bns

def edge_count(node):
  count = 0
  for p,c,v in edges(node):
    count += 1
    if isinstance(c, BaseLeaf):
      count += len(c.clt.parent)-1
  return count

model_dir = "models"
processed_data_dir = "Processed data"

datasets = benchmark_datasets + bns + healthcare_datasets

rows = []
for name in datasets:
  dataset_path = path.join(processed_data_dir, f"{name}.joblib")
  train, test, names, *_ = load(dataset_path)
    
  model_path = path.join(model_dir, f"{name}.joblib")
  try:
    cnets = load(model_path)
  except FileNotFoundError:
    continue
  scores = [[edge_count(row[0]), edge_count(row[2])] for row in cnets]
  edge_counts = np.mean(scores, axis = 0).round(2).tolist()
  scores = [[row[0].parameter_count, row[2].parameter_count] for row in cnets]
  rows.append([name] + edge_counts + np.mean(scores, axis = 0).round(2).tolist())
columns = ["Data set", "#E (LearnCNet)", "#E (KIL-CN)", "#P (LearnCNet)", "#P (KIL-CN)"]
df = pd.DataFrame(rows, columns = columns)
print (df.to_latex(float_format = "{0:,.2f}".format))