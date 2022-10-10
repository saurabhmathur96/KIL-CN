from itertools import product

import pandas as pd
import numpy as np
from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling

from pgmpy.models import BayesianNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from cnet2 import learn_cnet
from leaf2 import Dataset
from itertools import product

datasets = ["haberman", "cpu", "car", "abalone", "auto", "ljubljana", "redwine", "whitewine"]


def fetch_data(name: str, k: int):
  if name == "haberman":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    names = ["age", "year", "nodes", "survive"]
    frame = pd.read_csv(url, names=names)
    frame.survive = (frame.survive == 1).astype(int)
    for name in (col for col in frame.columns if col != "survive"):
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)

    data = frame[["age", "nodes", "year", "survive"]]

    X = data.to_numpy().astype(int)
    X = frame.to_numpy()
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()

  elif name == "redwine":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    frame = pd.read_csv(url, sep=";")

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
    frame = frame.drop(["vendor", "model"], axis=1)
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

    frame.drop(["breast", "quad"], axis=1, inplace=True)
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
    frame = pd.get_dummies(frame, columns=['class'])
    frame.drop(["name"], axis=1, inplace=True)

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
    frame["thal"].replace({3: 0, 6: 1, 7: 2}, inplace=True)
    frame["cp"] -= 1
    frame["slope"] -= 1
    frame["unhealthy"] = (frame.num.astype(int) != 0).astype(int)
    frame.drop(['num'], axis=1, inplace=True)
    for name in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
      frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
        .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
        .flatten().astype(int)
    for name in frame.columns:
      frame[name] = frame[name].astype(int)

    X = frame.to_numpy().astype(int)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
    r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
    return X_train, X_test, r, frame.columns.tolist()


def gen_data(N, n, r, C):
  X = np.random.uniform(0, 1, size=(N, n))
  for i in range(n):
    if np.count_nonzero(C[i]) == 0: continue

    # for i, j in zip(*np.nonzero(C)):
    X[:, i] = sum([np.exp(X[:, j] + 1) if C[i, j] == 1 else np.log(X[:, j] + 1) for j in np.flatnonzero(C[i])])

  for i in range(n):
    X[:, i] = KBinsDiscretizer(n_bins=r, encode='ordinal', strategy='kmeans') \
      .fit_transform(X[:, i].reshape(-1, 1)) \
      .flatten().astype(int)
  return X.astype(int)


"""



N, n, ri = 2000, 5, 3
X = np.random.uniform(0, ri, size = (N//2, n)).astype(int)
r = [ ri for i in range(n) ]
alpha = 1


C = np.zeros((n,n))
C[0, 1] = 1
C[0, 4] = 1
C[2, 3] = 1
eps = 0.001

D = Dataset(X, r)
epsilon, lambda_ = 0.001, 10
min_instances = 100
scope = list(range(n))
cnet = learn_cnet(D, scope, alpha, C, epsilon, lambda_, min_instances)
print (f"{cnet.loglik(X):.4f}")




Xt = gen_data(N, n, ri, C)
print (f"{cnet.loglik(Xt):.4f}")
"""


def fit_clt(frame: pd.DataFrame):
  est = TreeSearch(frame, root_node=frame.columns[0])
  dag = est.estimate(estimator_type="chow-liu", show_progress=False)
  model = BayesianNetwork(dag.edges())
  model.fit(frame, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=0.1)
  return model


def compute_monotonic_influences_(model: BayesianNetwork, frame: pd.DataFrame, r: list, sign: int, epsilon: float):
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


def compute_monotonic_influences(X, r, epsilon):
  n = len(r)
  alpha = 1

  node = learn_cnet(Dataset(X, r), list(range(n)), alpha)
  C = np.zeros((n, n), dtype=int)
  for i, j in product(range(n), range(n)):
    if i == j: continue
    for sign in [-1, +1]:
      d = node.delta(i, j, sign, epsilon)
      if np.all(d <= 0):
        C[i, j] = sign
  return C


def get_dataset(name: str, k: int = 3, noise: float = 0.3):
  X_train, X_test, r, names = fetch_data(name, k)
  n = len(r)
  if name == "cpu":
    epsilon = 0.1
  elif name == "car":
    epsilon = 0.005
  elif name == "auto":
    epsilon = 0.075
  elif name == "ljubljana":
    epsilon = 0.15
  elif name == "haberman":
    epsilon = 0.01
  elif name == "redwine":
    epsilon = 0.025
  elif name == "whitewine":
    epsilon = 0.025
  elif name == "abalone":
    epsilon = 0.05

  C = compute_monotonic_influences(X_train, r, epsilon)
  noise_size = int(len(X_train) * noise)
  n = len(r)

  np.random.shuffle(X_train)
  for i in np.flatnonzero(np.abs(C).sum(axis=1)):
    vals = sum(X_train[:noise_size, j] * r[i] / r[j] for j in np.flatnonzero(C[i])) / np.count_nonzero(C[i])
    noisy_vals = r[i] - 1 - np.ceil(vals).astype(int)
    X_train[:noise_size, i] = noisy_vals
  np.random.shuffle(X_train)

  return X_train, X_test, r, C, names


bns = ["sachs", "asia", "earthquake", "survey"]


def get_bn_dataset(name: str, noise: int = 0.3, N: int = 100):
  bn = BIFReader(f"{name}.bif").get_model()
  frame = BayesianModelSampling(bn).forward_sample(size=N)
  frame2 = BayesianModelSampling(bn).forward_sample(size=N)
  names = frame.columns.tolist()
  for col in names:
    frame[col] = frame[col].astype(int)
    frame2[col] = frame2[col].astype(int)

  n = len(frame.columns)
  if name == "sachs":
    r = [3 for _ in range(n)]
    epsilon = 0.025
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

  influences = compute_monotonic_influences_(bn, frame, r, +1, epsilon)
  for i, row in influences[influences.Degree != 0].iterrows():
    C[names.index(row.First), names.index(row.Second)] = +1

  influences = compute_monotonic_influences_(bn, frame, r, -1, epsilon)
  for i, row in influences[influences.Degree != 0].iterrows():
    C[names.index(row.First), names.index(row.Second)] = -1

  noise_size = int(noise * N)
  np.random.shuffle(X_train)
  for i in np.flatnonzero(np.abs(C).sum(axis=1)):
    vals = sum(X_train[:noise_size, j] * r[i] / r[j] for j in np.flatnonzero(C[i])) / np.count_nonzero(C[i])
    noisy_vals = r[i] - 1 - np.ceil(vals).astype(int)
    X_train[:noise_size, i] = noisy_vals
  np.random.shuffle(X_train)
  return X_train, X_test, r, C, names, bn


def format_influences(C, names):
  return [
    f"{names[j]} ≺ᴹ⁺ {names[i]}"
    if C[i, j] == +1
    else f"{names[j]} ≺ᴹ⁻ {names[i]}"
    for i, j in zip(*np.nonzero(C))
  ]
