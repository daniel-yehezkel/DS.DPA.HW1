import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from utils import load_folder

def main(train_path, model_path='models/final.joblib'):
    print("Training..")
    train_X, train_y = load_folder(train_path)

    # multiply class 1
    logic = train_y == 1
    new_X = [train_X] + [train_X[logic].copy() for _ in range (3)]
    new_y = [train_y] + [train_y[logic].copy() for _ in range (3)]

    train_X = pd.concat(new_X)
    train_y = pd.concat(new_y)
    
    models = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        RandomForestClassifier(min_samples_leaf=5),
        GradientBoostingClassifier(min_samples_leaf=5),
        MLPClassifier(),
        SGDClassifier(),
    ]
    ensembled = VotingClassifier(estimators=[(str(i), model) for i, model in enumerate(models)], voting='hard')
    ensembled.fit(X=train_X, y=train_y)
    dump(ensembled, model_path)


if __name__ == "__main__":
    train_path = 'data/train'
    main(train_path)