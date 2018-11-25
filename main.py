from neuralNetwork import NeuralNetwork
from util import Util
from movies import Movies
from sklearn import preprocessing
from pprint import pprint

#Busca a lista
moviesData = Util.getData()

#Criando os labels
actorLabel = preprocessing.LabelEncoder()
directorLabel = preprocessing.LabelEncoder()
genreLabel = preprocessing.LabelEncoder()

#Le as colunas selecionadas e cria os labels dos mesmo, transformando para int
actorLabel.fit(moviesData['actor'])
moviesData['actor'] = actorLabel.transform(moviesData['actor'])
directorLabel.fit(moviesData['director'])
moviesData['director'] = directorLabel.transform(moviesData['director'])
genreLabel.fit(moviesData['genre'])
moviesData['genre'] = genreLabel.transform(moviesData['genre'])

#Separamos os inputs do output
X = moviesData.iloc[:, 0:6]
y = moviesData['profit']

#1
trainSize = 0.8
neurons = [13,13,13]
iterations = 1000000

# for i in range(10):

#     X_train, X_test, y_train, y_test = NeuralNetwork.splitData(X, y, trainSize)

#     X_train, X_test = NeuralNetwork.normalizeData(X_train, X_test)

#     trainedAI = NeuralNetwork.trainingAI(X_train, y_train, neurons, iterations)

#     Util.saveTrainedAI(trainedAI, "neuralNetwork_" + str(i))

#     predictions = NeuralNetwork.predict(trainedAI, X_test)

#     result = NeuralNetwork.quadritc(y_test, predictions)

#     Util.saveResults(result, 1)

trainedAI = Util.openTrainedAI("neuralNetwork_" + "1")

actorList = Util.listAll(actorLabel)
genreList = Util.listAll(genreLabel)
directorList = Util.listAll(directorLabel)
pprint(actorList)
pprint("---------------------")
pprint(directorList)
pprint("---------------------")
pprint(genreList)
pprint("---------------------")


actorIndex = Util.findIndex(actorLabel, "Zoe Saldana")
pprint(actorIndex)
directorIndex = Util.findIndex(directorLabel, "Francis Ford Coppola")
pprint(directorIndex)
genreIndex = Util.findIndex(genreLabel, "Comedy")
pprint(genreIndex)

pred = Util.preparePredictions(1000000, 2000000, actorIndex, directorIndex, 120, genreIndex)
pprint(NeuralNetwork.predict(trainedAI, pred)[0])