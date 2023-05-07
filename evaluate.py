from joblib import load
from sklearn.metrics import f1_score
from utils import load_folder

def main(test_path, model_path='models/final.joblib'):
    print("Evaluating..")
    test_X, test_y = load_folder(test_path)
    model = load(model_path) 
    pred_test = model.predict(X=test_X)
    score = f1_score(test_y, pred_test)
    print(score)

if __name__ == "__main__":
    test_path = 'data/test'
    main(test_path)