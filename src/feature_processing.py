import pandas as pd
from src import config


def feature_process(jamones_calificar, score_jamonosidad):
    # jamones_calificar = pd.read_csv(config.JAMONES_CALIFICAR)
    # score_jamonosidad = pd.read_csv(config.SCORE_JAMONOSIDAD)

    # print(score_jamonosidad.head())
    # print(jamones_calificar.head())
    #
    # print(score_jamonosidad.info())
    # print(jamones_calificar.info())
    #
    # print(score_jamonosidad[['jamon', 'score', 'v1', 'v2', 'v3']].sort_values(by='score'))

    X_train, y_test = score_jamonosidad[['v1', 'v2', 'v3']], [int(x) for x in score_jamonosidad['score']]

    X_test = jamones_calificar[['v1', 'v2', 'v3']]

    return X_train, y_test, X_test