import pickle
import tqdm
import pandas as pd
import numpy as np

from os import path

from dataset import get_benchmark_dataset, benchmark_datasets
from dataset import get_healthcare_dataset, healthcare_datasets
from dataset import get_bn_dataset, bns

np.random.seed(0)
processed_data_dir = "Processed data"

print ("Preprocessing benchmark datasets")
for name in tqdm.tqdm(benchmark_datasets):
  train, test, names = get_benchmark_dataset(name)
  dataset_path = path.join(processed_data_dir, f"{name}.joblib")
  
  with open(dataset_path, "wb") as pfile:
    pickle.dump([train, test, names], pfile, protocol=pickle.HIGHEST_PROTOCOL)

print ("Preprocessing healthcare datasets")
for name in tqdm.tqdm(healthcare_datasets):
  train, test, names = get_healthcare_dataset(name)
  dataset_path = path.join(processed_data_dir, f"{name}.joblib")
  
  with open(dataset_path, "wb") as pfile:
    pickle.dump([train, test, names], pfile, protocol=pickle.HIGHEST_PROTOCOL)


print ("Processing bns")
for name in tqdm.tqdm(bns):
  train, test, names, bn = get_bn_dataset(name)
  dataset_path = path.join(processed_data_dir, f"{name}.joblib")
  
  with open(dataset_path, "wb") as pfile:
    pickle.dump([train, test, names, bn], pfile, protocol=pickle.HIGHEST_PROTOCOL)
