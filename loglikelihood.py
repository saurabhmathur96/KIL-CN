import numpy as np
import pandas as pd

from os import path
from joblib import load

from dataset import benchmark_datasets, healthcare_datasets, bns

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
  scores = [[cn.loglik(test) for cn in row] for row in cnets]
  rows.append([name] + np.mean(scores, axis = 0).round(2).tolist())

columns = ["Data set", "LearnCNet", "KIL-CN(-S)", "KIL-CN"]
df = pd.DataFrame(rows, columns = columns)
print (df.to_latex(float_format = "{0:,.2f}".format))