from itertools import product

import pandas as pd
import numpy as np
from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.model_selection import train_test_split

datasets = ["haberman", "cpu", "car", "abalone", "auto", "ljubljana", "yeast"]

def fetch_data(name: str, k:int):
  if name == "haberman":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    names = ["age", "year", "nodes", "survive"]
    frame = pd.read_csv(url, names = names)
    frame.survive = (frame.survive == 1).astype(int)
    frame.age = pd.cut(frame.age, k, labels = range(k))
    frame.year = pd.cut(frame.year, k, labels = range(k))
    frame.nodes = pd.cut(frame.nodes, k, labels = range(k))
    data = frame[["age", "nodes", "year", "survive"]]

    X = data.to_numpy().astype(int)
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return frame, r
  
  elif name == "cpu":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
    names = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    frame = pd.read_csv(url, names = names)
    frame = frame.drop(["vendor", "model"], axis=1)
    for name in frame.columns:
      frame[name] = pd.cut(frame[name], k, labels = range(k)).astype(int)

    X = frame.to_numpy()
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return frame, r
  
  elif name == "car":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    names = ["price", "maint", "doors", "person", "luggage", "safety", "class"]
    frame = pd.read_csv(url, names = names)
    frame.price.replace({ "vhigh": 3, "high": 2, "med": 1, "low": 0 }, inplace=True)
    frame.maint.replace({ "vhigh": 3, "high": 2, "med": 1, "low": 0 },  inplace=True)
    frame.doors.replace({ "2": 0, "3": 1, "4": 2, "5more": 3 },  inplace=True)
    frame.person.replace({"2": 0, "4": 1, "more": 2}, inplace=True)
    frame.luggage.replace({"small":0, "med":1, "big":2}, inplace=True)
    frame.safety.replace({"low":0, "med":1, "high":2}, inplace=True)
    frame["class"].replace({"unacc":0, "acc":1, "good":2, "vgood":3}, inplace=True)

    # for name in ["price", "maint", "doors", "person", "class"]:
    #   frame[name] = pd.cut(frame[name], k, labels = range(k)).astype(int)

    X = frame.to_numpy()
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]

    return frame, r
  
  elif name == "abalone":
    names = ["Sex", "Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell", "Rings"]
    frame = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", names = names)
    frame.Sex.replace({"I": 0, "M": 1, "F":2}, inplace=True)
    for name in frame.columns:
      if name not in ("Sex",):
        frame[name] = pd.cut(frame[name], k, labels = range(k)).astype(int)

    X = frame.to_numpy()
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]

    return frame, r

  elif name == "auto":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    names = ["mpg", "cylinders", "disp", "horsepwr", 
            "weight", "accel", "modelyear", "origin",
            "carname"]
    frame = pd.read_csv(url, names = names, sep = "\s+", na_values="?").dropna()
    for name in ["mpg", "disp", "horsepwr", "weight", "accel", "modelyear"]:
      frame[name] = pd.cut(frame[name], k, labels = range(k)).astype(int)
    frame.origin.replace({1:0, 2:1, 3:2},inplace=True)
    frame.cylinders.replace({3:0, 4:1, 5:2, 6:3, 8:4}, inplace=True)
    frame.drop(["carname"], axis = 1, inplace=True)

    X = frame.to_numpy()

    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return frame, r
  
  elif name == "ljubljana":
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
    names = ["class", "age", "menopause", "size", "invnodes", "nodecaps", "degmalig", "breast", "quad", "irradiat"]
    frame = pd.read_csv(url, names = names, na_values="?").dropna()
    frame["class"].replace(dict(zip(["no-recurrence-events", "recurrence-events"], range(2))), inplace=True)
    frame["age"].replace(dict(zip(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'], range(9))), inplace=True)
    frame["menopause"].replace(dict(zip(["premeno", "lt40", "ge40"], range(3))), inplace=True)
    frame["size"].replace(dict(zip(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'], range(12))), inplace=True)
    frame["invnodes"].replace(dict(zip(['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'], range(13))), inplace=True)
    frame["nodecaps"].replace({"no": 0, "yes": 1}, inplace=True)
    frame["degmalig"] -= 1
    frame["irradiat"].replace({"no":0,"yes":1}, inplace=True)

    for name in ["age", "size", "invnodes"]:
      frame[name] = pd.cut(frame[name], k, labels = range(k)).astype(int)

    frame.drop(["breast", "quad"], axis=1, inplace=True)
    X = frame.to_numpy()

    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return frame, r
  
  elif name == "yeast":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"

    names = ["name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]
    frame = pd.read_csv(url,  names = names, sep = "\s+", na_values = "?")
    class_names = ['CYT', 'NUC', 'MIT', 'ME3', 'ME2', 'ME1', 'EXC', 'VAC', 'POX', 'ERL']
    frame["class"].replace(dict(zip(class_names, range(10))), inplace=True)
    frame.drop(["name"], axis=1, inplace=True)

    frame.erl = (frame.erl == 1).astype(int)
    for name in ["mcg", "gvh", "alm", "mit", "pox", "vac", "nuc"]:
      frame[name] = pd.cut(frame[name], k, labels = range(k)).astype(int)
    X = frame.to_numpy()
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return frame, r

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


def gen_data(N, n, r, C):
  X = np.random.uniform(0, 1, size=(N,n))
  for i in range(n):
    if np.count_nonzero(C[i]) == 0: continue
    
  # for i, j in zip(*np.nonzero(C)):
    X[:, i] = sum([2*C[i, j]*X[:, j]  for j in np.flatnonzero(C[i])]) 

  for i in range(n):
    values = pd.qcut(X[:, i], r, labels = range(r))
    X[:, i] = values.to_numpy().astype(int)
  return X.astype(int)


Xt = gen_data(N, n, ri, C)
print (f"{cnet.loglik(Xt):.4f}")
"""


def fit_clt(frame: pd.DataFrame):
  est = TreeSearch(frame, root_node=frame.columns[0])
  dag = est.estimate(estimator_type="chow-liu", show_progress=False)
  model = BayesianNetwork(dag.edges())
  model.fit(frame, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=0.1)
  return model


def compute_monotonic_influence(model: BayesianNetwork, frame: pd.DataFrame, r: list, sign: int, epsilon: float):
  rows = []
  names = frame.columns.tolist()
  inference = VariableElimination(model)
  for first, second in product(frame.columns, frame.columns):
    numerator = inference.query([first, second], show_progress=False)
    denominator = inference.query([second], show_progress=False)
    frange = list(range(r[names.index(first)]))
    srange = list(range(r[names.index(second)]))
    terms = np.array([
      np.cumsum([numerator.get_value(**{first: fval, second: sval})/denominator.get_value(**{second: sval})
                 for fval in frange[:-1]])
      for sval in srange
    ]).T

    diffs = np.fromiter((
      sign * (row[vj1] - row[vj2])
      for row in terms
      for vj2, vj1 in product(srange, srange)
      if vj2 > vj1
    ), dtype=float)

    C = np.all((diffs + epsilon) > 0)
    degree = C*np.sum(diffs)/len(srange)
    rows.append((first, second, degree))

  return pd.DataFrame(rows, columns=("First", "Second", "Degree"))


def get_dataset(name: str, k: int = 3, noise: float = 0.2, epsilon: float = -0.001):
  frame, r = fetch_data(name, k)
  names = frame.columns.tolist()
  n = len(r)
  C = np.zeros((n, n))
  clt = fit_clt(frame)
  for sign in [-1, +1]:
    influences = compute_monotonic_influence(clt, frame, r, sign, epsilon)
    influences = influences[influences.Degree > 0]
    if len(influences) > 0:
      threshold = np.percentile(influences.Degree, 75)
      influences = influences[influences.Degree > threshold]
      for i, row in influences.sort_values(by="Degree", ascending=False).head(len(r) // 2).iterrows():
        C[names.index(row.First), names.index(row.Second)] = sign

  X = frame.to_numpy()
  X_train, X_test = train_test_split(X, test_size=0.5, random_state=0)
  k = int(len(X_train)*noise)
  n = X.shape[1]
  for i in range(n):
    X_train[:k, i] = np.random.uniform(0, r[i], size=k)
  np.random.shuffle(X_train)
  return X_train, X_test, r, C, names


def format_influences(C, names):
    return [
      f"{names[j]} ≺ᴹ⁺ {names[i]}"
      if C[i, j] == +1
      else f"{names[i]} ≺ᴹ⁻ {names[j]}"
      for i, j in zip(*np.nonzero(C))
    ]
