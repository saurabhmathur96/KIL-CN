import pandas as pd
import numpy as np

from itertools import product
from os import path

from itertools import product
from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from cnet2 import learn_cnet_base
from leaf2 import Dataset, DatasetWithKnowledge
from afe_dataset import read_afe_dataset

raw_data_dir = "Raw data"
bn_dir = "BNs"
benchmark_datasets = ["cpu", "ljubljana", "auto", "car", "abalone", "redwine", "whitewine", "yeast"]

def fetch_data(name: str, k: int):
  if name == "haberman":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    names = ["age", "year", "nodes", "survive"]
    frame = pd.read_csv(url, names=names)
    frame.survive = (frame.survive == 1).astype(int)
    frame.nodes = pd.cut(frame.nodes, [-np.inf, 10, 30, np.inf], labels = np.arange(3))
    frame.year = pd.cut(frame.year, [-np.inf, 60, 62, 64, 66, np.inf], labels = np.arange(5))
    frame.age = pd.cut(frame.age, [-np.inf, 40, 50, 60, 70, np.inf], labels = np.arange(5))
    #for name in (col for col in frame.columns if col != "survive"):
    #  frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
    #    .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
    #    .flatten().astype(int)

    data = frame[["age", "nodes", "year", "survive"]]

    X = data.to_numpy().astype(int)
    X = frame.to_numpy()
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()

  elif name == "redwine":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    frame = pd.read_csv(url, sep=";")
    # Fixed
    k = 2
    for name in frame.columns:
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)

    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()

  elif name == "whitewine":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    frame = pd.read_csv(url, sep=";")
    # Fixed
    k = 2
    for name in frame.columns:
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)

    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()


  elif name == "cpu":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
    names = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    frame = pd.read_csv(url, names=names)
    frame["logPRP"] = np.log(frame.PRP + 1e-6)
    frame = frame.drop(["vendor", "model", "PRP", "ERP"], axis=1)
    for name in frame.columns:
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)
    
    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()

  elif name == "car":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    names = ["price", "maint", "doors", "person", "luggage", "safety", "class"]
    frame = pd.read_csv(url, names=names)
    frame.price.replace({"vhigh": 3, "high": 2, "med": 1, "low": 0}, inplace=True)
    frame.maint.replace({"vhigh": 3, "high": 2, "med": 1, "low": 0}, inplace=True)
    frame.doors.replace({"2": 0, "3": 1, "4": 2, "5more": 3}, inplace=True)
    frame.person.replace({"2": 0, "4": 1, "more": 2}, inplace=True)
    frame.luggage.replace({"small": 0, "med": 1, "big": 2}, inplace=True)
    frame.safety.replace({"low": 0, "med": 1, "high": 2}, inplace=True)
    frame["class"].replace({"unacc": 0, "acc": 1, "good": 2, "vgood": 3}, inplace=True)

    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()

  elif name == "abalone":
    names = ["Sex", "Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell", "Rings"]
    frame = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", names=names)
    frame.Sex.replace({"I": 0, "M": 1, "F": 2}, inplace=True)
    for name in frame.columns:
      if name not in ("Sex",):
        frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
          .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
          .flatten().astype(int)

    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()

  elif name == "auto":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    names = ["mpg", "cylinders", "disp", "horsepwr",
             "weight", "accel", "modelyear", "origin",
             "carname"]
    frame = pd.read_csv(url, names=names, sep="\s+", na_values="?").dropna()
    for name in ["mpg", "disp", "horsepwr", "weight", "accel", "modelyear"]:
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)
    frame.origin.replace({1: 0, 2: 1, 3: 2}, inplace=True)
    frame.cylinders.replace({3: 0, 4: 1, 5: 2, 6: 3, 8: 4}, inplace=True)
    frame.drop(["carname"], axis=1, inplace=True)

    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, 0])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()

  elif name == "ljubljana":
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
    names = ["class", "age", "menopause", "size", "invnodes", "nodecaps", "degmalig", "breast", "quad", "irradiat"]
    frame = pd.read_csv(url, names=names, na_values="?").dropna()
    frame.drop(["quad"], axis = 1, inplace = True)
    frame["breast"] = (frame["breast"] == "left")
    frame["class"].replace(dict(zip(["no-recurrence-events", "recurrence-events"], range(2))), inplace=True)
    frame["age"].replace(
      dict(zip(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'], range(9))),
      inplace=True)
    frame["menopause"].replace(dict(zip(["premeno", "lt40", "ge40"], range(3))), inplace=True)
    frame["size"].replace(dict(
      zip(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'],
          range(12))), inplace=True)
    frame["invnodes"].replace(dict(zip(
      ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'],
      range(13))), inplace=True)
    frame["nodecaps"].replace({"no": 0, "yes": 1}, inplace=True)
    frame["degmalig"] -= 1
    frame["irradiat"].replace({"no": 0, "yes": 1}, inplace=True)
    
    for name in ["age", "size", "invnodes"]:
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)

    # frame.drop(["breast", "quad"], axis=1, inplace=True)
    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, 0])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()
  elif name == "yeast":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"

    names = ["name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]
    frame = pd.read_csv(url, names=names, sep="\s+", na_values="?")
    # class_names = ['CYT', 'NUC', 'MIT', 'ME3', 'ME2', 'ME1', 'EXC', 'VAC', 'POX', 'ERL']
    # frame["class"].replace(dict(zip(class_names, range(10))), inplace=True)
    # frame = pd.get_dummies(frame, columns=['class'])
    frame.drop(["name"], axis=1, inplace=True)
    frame["class"] = (frame["class"] == "CYT").astype(int)
    frame.erl = (frame.erl == 1).astype(int)
    for name in ["mcg", "gvh", "alm", "mit", "pox", "vac", "nuc"]:
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)

    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()

  elif name == "cleveland":
    names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
             "thal", "num"]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    frame = pd.read_csv(url, names=names, na_values="?").dropna()
    frame = frame[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "num", "thalach", "exang", "oldpeak", "slope"]]
    frame["unhealthy"] = (frame.num.astype(int) != 0).astype(int)
    frame.drop(['num'], axis=1, inplace=True)
    frame["chol"] = ((frame["chol"] < 200) | (frame["chol"] > 240)) # not normal
    frame["trestbps"] = pd.cut(frame["trestbps"], [0, 120, 140, np.inf], labels = np.arange(3))
    frame["restecg"] = (frame["restecg"] != 0) # not normal
    frame["cp"] = (frame["cp"] != 4) # chest pain present
    
    frame["age"] = pd.cut(frame["age"], [0, 40, 60, np.inf], labels = np.arange(3))
    
    for name in ["thalach", "oldpeak", "slope"]:
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)
      
    for name in frame.columns:
      frame[name] = frame[name].astype(int)
    
    
    
    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()
  
  elif name == "diabetes":
    frame = pd.read_csv(path.join(raw_data_dir, "diabetes", "diabetes.csv"))
    frame.iloc[:, [1,2,3,4,5,6,7]] = frame.iloc[:, [1,2,3,4,5,6,7]].replace(0, np.NaN)
    frame = frame[["age", "mass", "pedi", "pres", "class"]]
    frame = frame.dropna()
    frame["class"] = (frame["class"] == "tested_positive").astype(int)
    # frame["age"] = (frame["age"] >= 45).astype(int)
    # for name in ["preg", "plas", "pres", "skin", "insu", "mass", "pedi"]:
    #  frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
    #    .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
    #    .flatten().astype(int)
    """
    frame["age"] = pd.cut(frame["age"], [-np.inf, 25, 35, np.inf], labels = np.arange(3))
    # frame["preg"] = frame["preg"] >= 4  #pd.cut(frame["preg"], [-np.inf, 2, 6, np.inf], labels = np.arange(3))
    # frame["plas"] = frame["plas"] > 155 # pd.cut(frame["plas"], [-np.inf, 89, 107, 123, 143, np.inf], labels = np.arange(5))
    frame["pres"] = pd.cut(frame["pres"], [-np.inf, 76, 98, np.inf], labels = np.arange(3))
    # frame["skin"] = pd.cut(frame["skin"], [-np.inf, 25, 32, np.inf], labels = np.arange(3))
    # frame["insu"] = pd.cut(frame["insu"], [-np.inf, 75, 150, np.inf], labels = np.arange(3))
    frame["mass"] = pd.cut(frame["mass"], [-np.inf, 22.8, 26.8, 33.6, np.inf], labels = np.arange(4))
    frame["pedi"] = (frame["pedi"] > 0.525)
    """
    frame["age"] = pd.cut(frame["age"], [-np.inf, 30, 40, np.inf], labels = np.arange(3))
    frame["mass"] = pd.cut(frame["mass"], [-np.inf, 22.8, 26.8, 33.6, 35.6, np.inf], labels = np.arange(5))
    frame["pres"] = pd.cut(frame["pres"], [-np.inf, 76.1, 98.1, np.inf], labels = np.arange(3))
    frame["pedi"] = pd.cut(frame["pedi"], [-np.inf, .244, .525, .805, 1.11, np.inf], labels = np.arange(5))
    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()


def compute_monotonic_influences_from_cnet(node, epsilon):
  n = len(node.scope)
  C = np.zeros((n, n), dtype=int)
  for i, j in product(range(n), range(n)):
    if i == j: continue
    for sign in [-1, +1]:
      d = node.delta(i, j, sign, epsilon)
      if np.all(d <= 0):
        C[i, j] = sign
  return C


def get_benchmark_dataset(name: str):
  # ["cpu", "ljubljana", "haberman", "auto", "car", "abalone", "redwine", "whitewine", "yeast"]
  k = 3
  if k in ("redwine", "whitewine"):
    k = 2
  X_train, X_test, r, names = fetch_data(name, k)
  n = len(r)
  if name == "cpu":
    epsilon = 0.075
  elif name == "ljubljana":
    epsilon = 0.125
  # elif name == "haberman":
  #  epsilon = 0.01
  elif name == "auto":
    epsilon = 0.075
  elif name == "car":
    epsilon = 0.01
  elif name == "abalone":
    epsilon = 0.05
  elif name == "redwine":
    epsilon = 0.25
  elif name == "whitewine":
    epsilon = 0.25
  elif name == "yeast":
    epsilon = 0.05
    
  n = len(r)
  node = learn_cnet_base(Dataset(X_train, r), 1, min_variables = 5 if n > 5 else n-1)
  C = compute_monotonic_influences_from_cnet(node, epsilon)
  
  
  train = DatasetWithKnowledge(X_train, r, C, 0.001)
  test = Dataset(X_test, r)
  return train, test, names

healthcare_datasets = ["haberman", "cleveland", "diabetes", "ppd", "adni", "numom2b_mixed_a", "numom2b_mixed_b"]
def get_healthcare_dataset(name: str):
  if name == "haberman":
    X_train, X_test, r, names = fetch_data(name, 3)
    n = len(r)
    C = np.zeros((n, n))
    C[3, (0,2)]=-1
    C[3, 1] = +1
    
  elif name == "cleveland":
    X_train, X_test, r, names = fetch_data(name, 3)
    n = len(r)
    C = np.zeros((n, n))
    C[11, (0, 1,3,4,6)] = +1
    C[11, 2] = -1
  elif name == "diabetes":
    X_train, X_test, r, names = fetch_data(name, 3)
    n = len(r)
    C = np.zeros((n,n))
    # C[8, (0,6,7)] = +1
    # C[(1,2), 8] = +1
    # C[(4), 8] = -1
    C[-1, 0:-1] = +1
  elif name == "ppd":
    frame = read_afe_dataset("ppd")
    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    names = frame.columns.tolist()
    n = len(r)
    C = np.zeros((n,n))
    C[10, (4, 7, 8)] = +1
    C[10, (9)] = -1
  elif name == "adni":
    frame = read_afe_dataset("adni")
    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    names = frame.columns.tolist()
    n = len(r)
    C = np.zeros((n,n))
    C[6, (1,2)] = +1
    C[6, (0)] = -1
    
  elif name == "rare":
    frame = read_afe_dataset("rare")
    
    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    names = frame.columns.tolist()
    n = len(r)
    C = np.zeros((n,n))
    
    C[names.index('rare'), names.index('age')] = +1
    C[names.index('married'), names.index('age')] = +1
    C[ names.index('rare'), names.index('gender_m')] = -1
    # influences = [names.index(i) for i in ["health_video", "online_discuss", "review_hospital", "memorialize", "specialists"]]
    # C[influences, -1] = +1
  
  elif name == "numom2b_mixed_a":
    frame = pd.read_csv(path.join("Raw data", "nuMoM2b", f"{name}.csv"), index_col=0)
    X = frame.to_numpy().astype(int)
    r = (1+X.max(axis=0)).tolist()
    X_train, X_test = train_test_split(X, test_size = 0.5, stratify = X[:, 2], random_state = 0)
    names = frame.columns.tolist()
    n = len(r)
    C = np.zeros((n,n))
    C[2] = [1, -1, 0, 1, 1, 1, 1]

  elif name == "numom2b_mixed_b":
    frame = pd.read_csv(path.join("Raw data", "nuMoM2b", f"{name}.csv"), index_col=0)
    X = frame.to_numpy().astype(int)
    r = (1+X.max(axis=0)).tolist()
    X_train, X_test = train_test_split(X, test_size = 0.5, stratify = X[:, 3], random_state = 0)
    names = frame.columns.tolist()
    n = len(r)
    C = np.zeros((n,n))
    C[3] = [+1, -1, +1, 0, +1, +1, +1, +1]
  
  
  train = DatasetWithKnowledge(X_train, r, C, 0.001)
  test = Dataset(X_test, r)
  return train, test, names

def compute_monotonic_influences_from_bn(model: BayesianNetwork, frame: pd.DataFrame, r: list, sign: int, epsilon: float):
  rows = []
  names = frame.columns.tolist()
  inference = VariableElimination(model)
  for first, second in product(frame.columns, frame.columns):
    if first == second: continue
    numerator = inference.query([first, second], show_progress=False)
    denominator = inference.query([second], show_progress=False)
    frange = list(range(r[names.index(first)]))
    srange = list(range(r[names.index(second)]))
    terms = np.array([
      np.cumsum([numerator.get_value(**{first: fval, second: sval}) / denominator.get_value(**{second: sval})
                 for fval in frange[:-1]])
      for sval in srange
    ]).T
    #  first is influenced by second

    diffs = np.fromiter((
      sign * (row[vj2] - row[vj1])
      for row in terms
      for vj2, vj1 in product(srange, srange)
      if vj2 > vj1
    ), dtype=float)

    C = np.all((diffs) + epsilon < 0)
    degree = C * np.sum(diffs) / len(srange)
    rows.append((first, second, degree))

  return pd.DataFrame(rows, columns=("First", "Second", "Degree"))

bns = ["sachs", "asia", "earthquake", "survey"]
def get_bn_dataset(name: str, N: int = 100):
  bn_path = path.join(bn_dir, f"{name}.bif")
  bn = BIFReader(bn_path).get_model()
  frame = BayesianModelSampling(bn).forward_sample(size=N, show_progress = False, seed = 1234)
  frame2 = BayesianModelSampling(bn).forward_sample(size=N, show_progress = False, seed = 6789)
  names = frame.columns.tolist()
  for col in names:
    frame[col] = frame[col].astype(int)
    frame2[col] = frame2[col].astype(int)

  n = len(frame.columns)
  if name == "sachs":
    r = [3 for _ in range(n)]
    epsilon = 0.05
  elif name == "asia":
    r = [2 for _ in range(n)]
    epsilon = 0.75
  elif name == "earthquake":
    r = [2 for _ in range(n)]
    epsilon = 0.75
  elif name == "survey":
    r = [3, 2, 2, 2, 2, 3]
    epsilon = 0.05

  X_train = frame.to_numpy()
  X_test = frame2.to_numpy()
  C = np.zeros((n, n))

  influences = compute_monotonic_influences_from_bn(bn, frame, r, +1, epsilon)
  for i, row in influences[influences.Degree != 0].iterrows():
    C[names.index(row.First), names.index(row.Second)] = +1

  influences = compute_monotonic_influences_from_bn(bn, frame, r, -1, epsilon)
  for i, row in influences[influences.Degree != 0].iterrows():
    C[names.index(row.First), names.index(row.Second)] = -1
  
  
  train = DatasetWithKnowledge(X_train, r, C, 1e-3)
  test = Dataset(X_test, r)
  
  return train, test, names, bn



                              
def format_influences(C, names):
  return [
    f"{names[j]} ≺ᴹ⁺ {names[i]}"
    if C[i, j] == +1
    else f"{names[j]} ≺ᴹ⁻ {names[i]}"
    for i, j in zip(*np.nonzero(C))
  ]

