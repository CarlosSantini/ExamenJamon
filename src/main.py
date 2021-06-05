import pandas as pd
from src import config
from feature_processing import feature_process
from train import training


def save_calificaciones_csv(file, scores):
    # print(file.columns)
    file['score'] = scores
    file.to_csv(config.JAMONES_CALIFICAR, index=False)


if __name__ == "__main__":
    # Leer archivos csv
    jamones_calificar = pd.read_csv(config.JAMONES_CALIFICAR)
    score_jamonosidad = pd.read_csv(config.SCORE_JAMONOSIDAD)

    X_train, y_test, X_test = feature_process(jamones_calificar, score_jamonosidad)

    '''
    "DecisionTrees"; "RandomForest" 
    '''
    y_predict = training(X_train, y_test, X_test, 'RandomForest')

    save_calificaciones_csv(jamones_calificar, y_predict)

