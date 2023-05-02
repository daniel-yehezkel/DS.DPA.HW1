import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')


def load_patient(path):
    pdf = pd.read_csv(path, delimiter='|')
    label = int(len(pdf[pdf['SepsisLabel'] == 1]) > 0)

    if label:
        result = pd.concat((pdf[pdf['SepsisLabel'] == 0],  (pdf[pdf['SepsisLabel'] == 1]).iloc[0].to_frame().transpose()))
    else:
        result = pdf

    sample_dict = {}
    for col in result.columns:

        col_res = result[col].dropna()
        if len(col_res) > 0:
            value = col_res.iloc[-1]

            if col != "SepsisLabel":
                num_ups = ((col_res.values[:-1] - col_res.values[1:]) > 0).sum()
                num_downs = ((col_res.values[:-1] - col_res.values[1:]) < 0).sum()
                sample_dict[col + "__num_ups"] = num_ups / len(col_res)
                sample_dict[col + "__num_downs"] = num_downs / len(col_res)
        else:
            value = 0
        
        if col != "SepsisLabel" and (col + "__num_ups") not in sample_dict:
            sample_dict[col + "__num_ups"] = -1
            sample_dict[col + "__num_downs"] = -1
        
        sample_dict[col] = value
        
    return pd.Series(sample_dict).fillna(0)

def load_folder(path):
      patients_file_path = glob.glob(f"{path}/*")
      patients_psr = [load_patient(p) for p in patients_file_path]
      full_data = pd.concat(patients_psr, axis=1).transpose()
      
      if 0 in full_data.columns:
        full_data = full_data.drop(0, axis=1)
      elif '0'  in full_data.columns:
        full_data = full_data.drop('0', axis=1)

      X = full_data.drop(['SepsisLabel', 'EtCO2', 'Bilirubin_direct', 'TroponinI'], axis=1)
      y = full_data['SepsisLabel']

      return X, y

def main():
    print("hello")
    train_X, train_y = load_folder('data/train')
    train_X.to_csv('train_x.csv', index=False, header=True)
    train_y.to_csv('train_y.csv', index=False, header=True)

    # multiply class 1
    logic = train_y == 1
    new_X = [train_X] + [train_X[logic].copy() for _ in range (3)]
    new_y = [train_y] + [train_y[logic].copy() for _ in range (3)]

    train_X = pd.concat(new_X)
    train_y = pd.concat(new_y)

    test_X, test_y = load_folder('data/test')
    
    test_X.to_csv('test_x.csv', index=False, header=True)
    test_y.to_csv('test_y.csv', index=False, header=True)
    exit(0)
    scores = []
    for seed in [20]: # 42, 10, 20, 30, 40]:
        models = [
            RandomForestClassifier(random_state=seed),
            GradientBoostingClassifier(random_state=seed),
            AdaBoostClassifier(random_state=seed),
            RandomForestClassifier(min_samples_leaf=5, random_state=seed),
            GradientBoostingClassifier(min_samples_leaf=5, random_state=seed),
            MLPClassifier(random_state=seed),
            SGDClassifier(random_state=seed),
        ]
        ensembled = VotingClassifier(estimators=[(str(i), model) for i, model in enumerate(models)], voting='hard')
        ensembled.fit(X=train_X, y=train_y)
        pred_test = ensembled.predict(X=test_X)

        curr_score = f1_score(test_y, pred_test)
        scores.append(curr_score)
    
    final_score = np.mean(scores)
    print("All scores:", scores)
    print("Final Score:", final_score)
    print("Done")

if __name__ == "__main__":
    main()