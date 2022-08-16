import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

def fetch_data(name):
  if name == "haberman":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
    names = ["age", "year", "nodes", "survive"]
    frame = pd.read_csv(url, names = names)
    frame.survive = (frame.survive == 1)
    frame.age = pd.cut(frame.age, 3, labels = range(3))
    frame.year = pd.cut(frame.year, 3, labels = range(3))
    frame.nodes = pd.cut(frame.nodes, 3, labels = range(3))
    data = frame[["age", "nodes", "year", "survive"]]
    n = 4
    C = np.zeros((n, n))

    C[3, 1] = C[1, 3] = -1
    C[2, 0] = C[0, 2] = +1

    X = data.to_numpy().astype(int)
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return X, r, C 
  
  elif name == "cpu":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
    names = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    frame = pd.read_csv(url, names = names)
    frame = frame.drop(["vendor", "model"], axis=1)
    for name in frame.columns:
      frame[name] = pd.cut(frame[name], 3, labels = range(3)).astype(int)

    X = frame.to_numpy()
    n = 8
    C = np.zeros((n, n))
    C[7, 6] = C[7, 2] = 1
    C[5, 6] = C[5, 7] = 1
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return X, r, C
  
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

    for name in ["price", "maint", "doors", "person", "class"]:
      frame[name] = pd.cut(frame[name], 3, labels = range(3)).astype(int)

    X = frame.to_numpy()
    n = 7
    C = np.zeros((n, n))
    C[4, 6] = C[2, 6] = 1
    C[6, 3] = +1
    C[6, 1] = -1
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]

    return X, r, C
  
  elif name == "abalone":
    names = ["Sex", "Length", "Diameter", "Height", "Whole", "Shucked", "Viscera", "Shell", "Rings"]
    frame = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", names = names)
    frame.Sex.replace({"I": 0, "M": 1, "F":2}, inplace=True)
    for name in frame.columns:
      if name not in ("Sex",):
        frame[name] = pd.cut(frame[name], 3, labels = range(3)).astype(int)

    X = frame.to_numpy()
    n = 9
    C = np.zeros((n, n))
    C[2, 1] = C[1, 2] = +1
    C[5, 6] = C[6, 5] = +1
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]

    return X, r, C

  elif name == "auto":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    names = ["mpg", "cylinders", "disp", "horsepwr", 
            "weight", "accel", "modelyear", "origin",
            "carname"]
    frame = pd.read_csv(url, names = names, sep = "\s+", na_values="?").dropna()
    for name in ["mpg", "disp", "horsepwr", "weight", "accel", "modelyear"]:
      frame[name] = pd.cut(frame[name], 3, labels = range(3)).astype(int)
    frame.origin.replace({1:0, 2:1, 3:2},inplace=True)
    frame.cylinders.replace({3:0, 4:1, 5:2, 6:3, 8:4}, inplace=True)
    frame.drop(["carname"], axis = 1, inplace=True)

    X = frame.to_numpy()
    n = 8
    C = np.zeros((n, n))
    C[2, 4] = C[4, 2] = C[2, 3] = 1
    C[0, 4] = C[2, 5] = C[7, 4] = -1
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return X, r, C
  
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
      frame[name] = pd.cut(frame[name], 5, labels = range(5)).astype(int)

    frame.drop(["breast", "quad"], axis=1, inplace=True)
    X = frame.to_numpy()
    n = 8
    C = np.zeros((n, n))
    C[4, 5] = C[7, 3] = C[6, 5] = +1
    C[1, 0] = -1
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return X, r, C
  
  elif name == "yeast":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"

    names = ["name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]
    frame = pd.read_csv(url,  names = names, sep = "\s+", na_values = "?")
    class_names = ['CYT', 'NUC', 'MIT', 'ME3', 'ME2', 'ME1', 'EXC', 'VAC', 'POX', 'ERL']
    frame["class"].replace(dict(zip(class_names, range(10))), inplace=True)
    frame.drop(["name"], axis=1, inplace=True)

    frame.erl = (frame.erl == 1).astype(int)
    for name in ["mcg", "gvh", "alm", "mit", "pox", "vac", "nuc"]:
      frame[name] = pd.cut(frame[name], 3, labels = range(3)).astype(int)
    X = frame.to_numpy()
    n = 9
    C = np.zeros((n, n))
    C[8, 4] = C[0, 1] = C[1, 0] = 1
    r = [ (m + 1) for i, m in enumerate(X.max(axis = 0)) ]
    return X, r, C

def get_dataset(name, noise = 0.2):
  X, r, C = fetch_data(name)
  X_train, X_test = train_test_split(X, test_size = 0.5, random_state = 0)
  k = int(len(X_train)*noise)
  n = X.shape[1]
  for i in range(n):
    X_train[:k, i] = np.random.uniform(0, r[i], size = k)
  np.random.shuffle(X_train)
  return X_train, X_test, r, C 

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

