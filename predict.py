import glob
import pandas as pd


def load_patient(path):
    pdf = pd.read_csv(path, delimiter='|')
    label = int(len(pdf[pdf['SepsisLabel'] == 1]) > 0)
    
    if label:
        result = (pdf[pdf['SepsisLabel'] == 1]).iloc[0]
    else:
        result = pdf.iloc[-1]

    return result.fillna(0)
    # if label:
    #     result = pd.concat((pdf[pdf['SepsisLabel'] == 0],  (pdf[pdf['SepsisLabel'] == 1]).iloc[0]))
    # else:
    #     result = pdf

    # sample_dict = {}
    # for col in result.columns:
    #     col_res = result[col].dropna()
    #     if len(col_res) > 0:
    #         value = col_res.iloc[-1]
    #     else:
    #         value = 0
    #     sample_dict[col] = value

    # return pd.Series(sample_dict).fillna(0)

def load_folder(path):
      patients_file_path = glob.glob(f"{path}/*")
      patients_psr = [load_patient(p) for p in patients_file_path]
      full_data = pd.concat(patients_psr, axis=1).transpose()
      
      if 0 in full_data.columns:
        full_data = full_data.drop(0, axis=1)
      elif '0'  in full_data.columns:
        full_data = full_data.drop('0', axis=1)

      X = full_data.drop('SepsisLabel', axis=1)
      y = full_data['SepsisLabel']

      return X, y

def main():
    train_X, train_y = load_folder('data/train')
    test_X, test_y = load_folder('data/test')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    
    model = RandomForestClassifier()
    model.fit(train_X, train_y)
    pred_test = model.predict(test_X)

    score = f1_score(test_y, pred_test)
    print(score)

if __name__ == "__main__":
    main()