import pandas as pd
import numpy as np

from os import path
from itertools import chain

def read_afe_dataset(name: str):
  if name == "adni":
    data_path = path.join("Raw data", "ADNI", "Alz_clean_full_bkp.csv")
    data = pd.read_csv(data_path, index_col = "RID")

    demo_path = path.join("Raw data", "ADNI", "Sriraam400_demographics.csv")
    demo = pd.read_csv(demo_path, index_col = "RID")

    data = data.join(demo, how = "inner")[["MMSCORE", "AGE", "PTGENDER", "PTHAND", "PTEDUCAT", "PTRACCAT", "DXCURREN"]]
    data["AGE"] = pd.cut(data.AGE, [0, 65, 75, 85, np.inf], labels = np.arange(4))
    data["MMSCORE"] = pd.cut(data.MMSCORE, [0, 18, 23, np.inf], labels = np.arange(3))
    data["PTEDUCAT"] = pd.cut(data["PTEDUCAT"], [0, 12, 14, 16, 18, np.inf], labels = np.arange(5))
    data["PTGENDER"] = (data["PTGENDER"] == 1).astype(int)
    data["PTHAND"] = (data["PTHAND"] == 1).astype(int)
    data["PTRACCAT"].replace({2: 0, 4:1, 5:2 }, inplace = True)
    return data

  elif name == "ppd":
    data_path = path.join("Raw data", "PPD", "train_1.csv")
    data = pd.read_csv(data_path).dropna()
    data_path = path.join("Raw data", "PPD", "test_1.csv")
    data2 = pd.read_csv(data_path).dropna()
    
    data = pd.concat([data, data2])

    columns = list(chain.from_iterable([
        ["age", "citizen", "first_time_mom", "employed"],
        ["unplanned_1", "unplanned_2"],
        ["history_depression_1", "prenatal_depression_1"],
        ["life_stress"],
        ["maternity_blues"],
        ["child_care_stress_2", "child_care_stress_3"],
        ["marital_satisfaction_1", "marital_satisfaction_2", "marital_satisfaction_3"],
        ["social_support_1", "social_support_2", "social_support_3", "social_support_4"],
        ["class label -PPD"]
    ]))

    data = data[columns]
    data["age"] = pd.cut(data["age"], [0, 3, 4, 7], labels = np.arange(3))
    data["unplanned"] = data[["unplanned_1", "unplanned_2"]].any(axis = 1).astype(int)
    data["past_depression"] = data[["history_depression_1", "prenatal_depression_1"]].any(axis = 1).astype(int)
    data["child_care_stress"] = data[["child_care_stress_2", "child_care_stress_3"]].any(axis = 1).astype(int)
    data["partner_support"] = data[[ "marital_satisfaction_2", "marital_satisfaction_3", "social_support_1", "social_support_2", "social_support_3", "social_support_4"]].all(axis=1).astype(int)
    data["partner_support"] = (1-data["marital_satisfaction_1"])*data["partner_support"].astype(int)
    data["ppd"] = data["class label -PPD"]

    data.drop(
        ["unplanned_1", "unplanned_2"] + ["history_depression_1", "prenatal_depression_1"] + ["child_care_stress_2", "child_care_stress_3"]
    , axis = 1, inplace=True)

    data.drop(
        ["marital_satisfaction_1", "marital_satisfaction_2", "marital_satisfaction_3"] + ["social_support_1", "social_support_2", "social_support_3", "social_support_4"]
    , axis = 1, inplace=True)
    data.drop(
        ["class label -PPD"]
    , axis = 1, inplace=True)
    return data

  elif name == "rare":

    data_path = path.join("Raw data", "Rare", "rare_disease.CSV")
    data = pd.read_csv(data_path, encoding = "latin1").dropna()
    data["Rare"] = (data["Disease Type"] == 'R') | (data["Disease Type"] == 'B')
    data["Chronic"] = (data["Disease Type"] == 'C') | (data["Disease Type"] == 'B')
    data.drop(['Disease Type'], axis = 1, inplace=True)
    data["Employment"] = (data["Employment"] == "Full time").astype(int)
    data["Education"] = pd.cut(data["Education"], [0,5,10,np.inf], labels=np.arange(3)).astype(int)
    data["Gender:"] = (data["Gender:"] == "M").astype(int)
    data["Age:"] = pd.cut(data["Age:"], [0, 35, 45, 55, np.inf], labels = np.arange(4, dtype = int))
    data["Country of Residence:"] = (data["Country of Residence:"] == 'USA').astype(int)
    data["Marital Status:"] = (data["Marital Status:"] == "Married").astype(int)
    cols = ['Age:',
     'Gender:',
     'Country of Residence:',
     'Marital Status:',
     'Education',
     'Employment',
     'Chronic',
     'Rare']
    
    data = data[cols]
    data.columns = ['age', 'gender_m', 'usa', 'married', 'education', 'employed', 'rare']
    return data.astype(int)
        