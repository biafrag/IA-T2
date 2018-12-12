import carEvaluationProblem as p
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import time

class Knn(object):
	
	def __init__(self):
		self.accuracy = 0
		self.problem = p.CarProblem()
		np.random.seed(self.problem.seed)
		self.dataframe = self.problem.dataframe
		self.dataset = self.dataframe.values
	
	def trainKnn(self):
		# Criando a Knn, testando o melhor numero de vizinhos para testar
		bestK = 1
		bestAccuracy = 0
		for i in range(3,13,2):
			start = time.time()
			#criando o knn
			knn = KNeighborsClassifier(n_neighbors=i)
			# passando os dados de treino para o knn
			knn.fit(self.problem.X_Train, self.problem.Y_Train)
			print('Tempo de treino para k = {}: {} s'.format(i,time.time()-start))
			# Realizando a classificação com dados não usados no treino (dados de teste)
			predictions = knn.predict(self.problem.X_Test)
			start = time.time()
			#comparando os resultados entre a classificação 'real' e o que a knn classificou para os casos de teste
			score = accuracy_score(self.problem.Y_Test, predictions)
			print('Tempo de classificaçao dos testes para k = {}: {} s'.format(i,time.time()-start))
			print('score', score,' k ', i)
			if score > bestAccuracy:
				bestAccuracy = score
				bestK = i
		self.k = bestK
		self.accuracy = bestAccuracy

	def printResults(self):
		print('melhor resultado para a Knn: Score = {}, Numero de vizinhos = {}'.format(self.accuracy,self.k))

	def showDataHist(self):
		# Mostrando a distribuição dos atributos e classes
		self.problem.showDataHist()



if __name__ == '__main__':
	k = Knn()
	k.trainKnn()
	k.printResults()
	k.showDataHist()

