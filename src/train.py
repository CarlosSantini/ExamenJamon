from src import config
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from xgboost import XGBClassifier


def training(X_train, y_train, X_test, select_model):

    MODELS = {
        "DecisionTrees": DecisionTreeClassifier(max_depth=5),
        "RandomForest": RandomForestClassifier(max_depth=None, n_estimators=10, max_features=1)
    }

    model = MODELS[select_model]

    model.fit(X_train, y_train)

    # y_predict = model.predict(X_train)
    # print("Predicciones con dataset de entrenamiento:\n"+y_predict)
    # print(accuracy_score(y_train, y_predict))

    y_predict = model.predict(X_test)
    # print("Predicciones con dataset de prueba:\n" , y_predict)

    return y_predict





