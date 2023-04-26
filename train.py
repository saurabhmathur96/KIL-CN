import copy
import argparse
import pickle
import numpy as np
import pandas as pd

from os import path
from joblib import load, Parallel, delayed

from dataset import format_influences
from dataset import benchmark_datasets, healthcare_datasets, bns
from cnet2 import learn_cnet_base, learn_cnet_with_knowledge

np.random.seed(0)
alpha = 1
repeat = 10
tries = 10
model_dir = "models"
processed_data_dir = "Processed data"
total = len(benchmark_datasets) + len(healthcare_datasets) + len(bns)

parser = argparse.ArgumentParser()
parser.add_argument("dataset_type", type=str, choices=("benchmark", "healthcare", "bn"))
parser.add_argument("start", type=int, default=0)
parser.add_argument("end", type=int, default=-1)
args = parser.parse_args()

dataset_type = args.dataset_type
start = args.start
end = args.end 

datasets = { "benchmark": benchmark_datasets, 
             "healthcare": healthcare_datasets,
             "bn": bns }[dataset_type]
if end == -1:
  end = len(datasets)
  
with Parallel(n_jobs=-1, verbose=10) as parallel:
  for name in datasets[dataset_type][start:end]: 
    dataset_path = path.join(processed_data_dir, f"{name}.joblib")
    train, test, names, *_ = load(dataset_path)
    if name not in ("numom2b_mixed_a", "numom2b_mixed_b", "adni", "ppd", "rare"):
      train = train.add_noise(noise = 0.3) 
    samples = train.bootstrap_samples(repeat)
    
    n = len(names)
    def fit_models(i):
      sample = samples[i]
      node1 = learn_cnet_base(sample, alpha, min_variables = 5 if n > 5 else n-1)
      
      node2 = copy.deepcopy(node1)
      node2.fit_leaves_with_knowledge(sample, alpha = alpha, tries = tries)
      
      node3 = learn_cnet_with_knowledge(sample, alpha = alpha, tries = tries, min_variables = 5 if n > 5 else n-1)
      
      return [node1, node2, node3]
    
    result = parallel([delayed(fit_models)(i) for i in range(repeat)])
    
    model_path = path.join(model_dir, f"{name}.joblib")
    with open(model_path, "wb") as pfile:
      pickle.dump(result, pfile, protocol=pickle.HIGHEST_PROTOCOL)
