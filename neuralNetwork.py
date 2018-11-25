from util import Util
from pprint import pprint  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib 

class NeuralNetwork:

    #Divide os dados entre treino e teste
    @staticmethod
    def splitData(X, y, train_size):
        return train_test_split(X, y, train_size = train_size)
    
    #Normaliza os dados de treinamento
    @staticmethod
    def normalizeData(X_train, X_test):
        scaler = StandardScaler()
        
        #Encontra a media e padroniza com base nos dados passado
        scaler.fit(X_train)
        StandardScaler(copy=True, with_mean=True, with_std=True)

        #Centraliza e escala os dados com base na media encontrada no metodo fit
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)
        return x_train, x_test

    #Treina a rede neural
    @staticmethod
    def trainingAI(X_train, y_train, neurons, iterations):
        #Cria as camadas intermedierias com base no array de neurons
        #passado por parametro
        #Numero de iteracoes tambem passada por parametro
        mlp = MLPClassifier(hidden_layer_sizes=tuple(neurons),max_iter=iterations)
        return mlp.fit(X_train,y_train)

    #Metodo de predicao onde e passada a rede e os dados de teste
    @staticmethod
    def predict(mlp, X_test):
        return mlp.predict(X_test)

    #Acha o erro medio quadratico
    @staticmethod
    def quadritc(y_test, predictions):
        return mean_squared_error(y_test, predictions)
