import numpy as np
import pandas as pd

from os import path
from joblib import load

from dataset import benchmark_datasets, healthcare_datasets, bns, format_influences

processed_data_dir = "Processed data"

datasets = benchmark_datasets + bns + healthcare_datasets
rows = []
for name in datasets:
  dataset_path = path.join(processed_data_dir, f"{name}.joblib")
  train, test, names, *_ = load(dataset_path)
  print (name, format_influences(train.C, names))
  rows.append([name, train.X.shape[1], 
               len(format_influences(train.C, names)), 
               train.X.shape[0], test.X.shape[0]])
  
columns = ["name", "n_variables", "n_influences", "train_size", "test_size"]  
df = pd.DataFrame(rows, columns = columns)
# TODO: Re-run car
print (df.to_latex(index = False, float_format = "{0:,.2f}".format))