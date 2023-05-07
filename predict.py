import sys
from joblib import load
from utils import load_folder
import csv

def main(test_path, model_path='models/final.joblib'):
    
    test_X, _ = load_folder(test_path)
    model = load(model_path) 
    pred_test = model.predict(X=test_X)

    with open('prediction.csv', 'w', newline='') as pred_file:
       writer = csv.writer(pred_file)
       rows = [['id', 'prediction']]
       rows = rows + [[f'patient_{idx}', int(p)] for idx, p in enumerate(pred_test)]
       writer.writerows(rows)
     
    
if __name__ == "__main__":
    test_path = sys.argv[1]
    main(test_path)