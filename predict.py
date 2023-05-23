import sys
from joblib import load
from utils import load_folder
import csv
import glob

def main(test_path, model_path='models/final.joblib'):
    
    test_X, _ = load_folder(test_path)
    model = load(model_path) 
    pred_test = model.predict(X=test_X)

    patients_file_path = glob.glob(f"{test_path}/*")
    patients_file_path = sorted(patients_file_path, key=lambda x: int(x.split('_')[-1].split(".")[0]))
    patients_ids = [x.split("/")[1].split(".")[0] for x in patients_file_path]

    with open('prediction.csv', 'w', newline='') as pred_file:
       writer = csv.writer(pred_file)
       rows = [['id', 'prediction']]
       rows = rows + [[id, int(p)] for id, p in zip(patients_ids, pred_test)]
       writer.writerows(rows)
     
    
if __name__ == "__main__":
    test_path = sys.argv[1]
    main(test_path)