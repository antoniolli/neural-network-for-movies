from neuralNetwork import NeuralNetwork
from util import Util
from movies import Movies
from sklearn import preprocessing

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


#pprint(y.head())
#pprint(X['actor'])
#pprint(moviesData.head())
#pprint(actorLabel.classes_[0])
#pprint(list(genreLabel.inverse_transform([1, 2])))
#pprint(classification_report(y_test, predictions, target_names=target_names))

#Separamos os inputs do output
X = moviesData.iloc[:, 0:8]
y = moviesData['profit']

X_train, X_test, y_train, y_test = NeuralNetwork.splitData(X, y)

X_train, X_test = NeuralNetwork.normalizeData(X_train, X_test)

trainedAI = NeuralNetwork.trainingAI(X_train, y_train)

Util.saveTrainedAI(trainedAI, "teste_um")

predictions = NeuralNetwork.predict(trainedAI, X_test)

NeuralNetwork.quadritc(y_test, predictions)