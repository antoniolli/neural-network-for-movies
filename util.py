import csv
import pandas as pd
import numpy as np
from sklearn.externals import joblib 

class Util:
    #Le e transforma o CSV para um DataFrame, uma lista
    @staticmethod
    def getData():
        moviesData = pd.read_csv("./movies.csv", na_values = ['no info', '.'])
        return pd.DataFrame(moviesData)
    
    #Abre rede neural ja treinada previamente
    @staticmethod
    def openTrainedAI(name):
        return joblib.load(name + '.joblib')

    #Salva rede neural treinada
    @staticmethod
    def saveTrainedAI(trainedAI, name):
        joblib.dump(trainedAI, name + '.joblib')

    #Salva os resultados da automacao em um arquivo de texto
    @staticmethod
    def saveResults(result, i):
        f = open("results/results" + str(i) + '.txt', "a+")
        f.write(str(result) + "\n")
        f.close()

    @staticmethod
    def findIndex(label, value):
        index = np.where(label.classes_==value)
        return index[0][0]

    @staticmethod
    def listAll(label):
        np.set_printoptions(threshold=np.nan)
        allLabels = np.array(label.classes_)
        return allLabels

    @staticmethod
    def preparePredictions(budget, revenue, actor, director, runtime, genre):
        a = np.array([budget, revenue, actor, director, runtime, genre])
        b = np.reshape(a, (-1, 6))
        return b
