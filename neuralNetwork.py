from util import Util
from pprint import pprint  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib 

class NeuralNetwork:

    @staticmethod
    def splitData(X, y):
        return train_test_split(X, y)
        
    @staticmethod
    def normalizeData(X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        StandardScaler(copy=True, with_mean=True, with_std=True)
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)
        return x_train, x_test


    @staticmethod
    def trainingAI(X_train, y_train):
        mlp = MLPClassifier(hidden_layer_sizes=(3,2,5),max_iter=10000000)
        return mlp.fit(X_train,y_train)

    @staticmethod
    def predict(mlp, X_test):
        return mlp.predict(X_test)

    @staticmethod
    def quadritc(y_test, predictions):
        pprint(mean_squared_error(y_test, predictions))
