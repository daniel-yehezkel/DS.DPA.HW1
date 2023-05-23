import pandas as pd
import glob


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

            if col in ['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp', 'Unit1', 'Unit2']:
                num_ups = ((col_res.values[:-1] - col_res.values[1:]) > 0).sum()
                num_downs = ((col_res.values[:-1] - col_res.values[1:]) < 0).sum()
                sample_dict[col + "__num_ups"] = num_ups / len(col_res)
                sample_dict[col + "__num_downs"] = num_downs / len(col_res)
        else:
            value = 0
        
        if col in ['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp', 'Unit1', 'Unit2'] and (col + "__num_ups") not in sample_dict:
            sample_dict[col + "__num_ups"] = -1
            sample_dict[col + "__num_downs"] = -1
        
        sample_dict[col] = value
        
    return pd.Series(sample_dict).fillna(-1)


def load_folder(path):
      patients_file_path = glob.glob(f"{path}/*")
      patients_file_path = sorted(patients_file_path, key=lambda x: int(x.split('_')[-1].split(".")[0]))
      patients_psr = [load_patient(p) for p in patients_file_path]
      full_data = pd.concat(patients_psr, axis=1).transpose()
      
      if 0 in full_data.columns:
        full_data = full_data.drop(0, axis=1)
      elif '0'  in full_data.columns:
        full_data = full_data.drop('0', axis=1)

      X = full_data.drop(['SepsisLabel', 'EtCO2', 'Bilirubin_direct', 'TroponinI'], axis=1)
      y = full_data['SepsisLabel']

      return X, y