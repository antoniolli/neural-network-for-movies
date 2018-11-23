import csv
import pandas as pd
from sklearn.externals import joblib 

class Util:
    #Le e transforma o CSV para um DataFrame, uma lista
    @staticmethod
    def getData():
        moviesData = pd.read_csv("./movies.csv", na_values = ['no info', '.'])
        return pd.DataFrame(moviesData)
    
    @staticmethod
    def openTrainedAI(name):
        joblib.load(name + '.joblib')

    @staticmethod
    def saveTrainedAI(trainedAI, name):
        joblib.dump(trainedAI, name + '.joblib')

