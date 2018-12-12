from sklearn import svm
from sklearn.svm import SVC

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import carEvaluationProblem as p

import time

class SVM(object):	
    def __init__(self):
        self.accuracy = 0
        self.problem = p.CarProblem()
        np.random.seed(self.problem.seed)
        self.dataframe = self.problem.dataframe
        self.dataset = self.dataframe.values
        
    # Fazendo a standartização dos dados ler o site a seguir para explicação:
    # Obtivemos resultados melhores ulizando esse passo
    #  https://towardsdatascience.com/effect-of-feature-standardization-on-linear-support-vector-machines-13213765b812
    def standartizeTrainData(self):   
        sc = StandardScaler()
        sc.fit(self.problem.X_Train)
        X_train_std=sc.transform(self.problem.X_Train)
        X_test_std=sc.transform(self.problem.X_Test)
        return X_test_std, X_train_std

    def standartize(self,data):
        sc = StandardScaler()
        sc.fit(data)
        data_std=sc.transform(data)        
        return data_std

    def trainSVM(self):
        start = time.time()
        X_test_std,X_train_std = self.standartizeTrainData()
        # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
        # kernel é o tipo de formulação matematica do svm, ebf é "Radial Basis Function"
        # exp( -gamma * (|| x - x' ||)^2 )
        # O parametro C quanto 
        self.svc = svm.SVC(kernel='rbf', C = 7,gamma = 0.4)
        self.svc.fit(X_train_std,self.problem.Y_Train)
        print('tempo de treino: {}s'.format(time.time()-start))
        start = time.time()
        s_pred=self.svc.predict(X_test_std)
        print('tempo de classificação dos casos de teste: {}s'.format(time.time()-start))
        self.accuracy = accuracy_score(self.problem.Y_Test,s_pred)
        self.errosSum = (self.problem.Y_Test!=s_pred).sum()
        
        

    def printResults(self):
        # imprimindo o numero de elementos classificados errados
        print('Misclassified samples using SVM are: {}'.format(self.errosSum))
        # imprimindo a acuracia
        print('Classification Accuracy of SVM is {} '.format(self.accuracy))

    def classify(self, X_elements ): 
        ''' Faz a estandartização dos dados e faz a classificação,
            Recebe uma matriz com cada linha contendo os valores de cada atributo
            **Lembrando que os atributos foram substituidos pro numeros '''
        X_elements_std = self.standartize(X_elements)
        return self.svc.predict(X_elements_std)


if __name__ == '__main__':
    s = SVM()
    s.trainSVM()
    s.printResults()
    #print(s.classify([[1,2,1,3,1,3],[2,2,3,2,1,1],[4,1,4,1,2,2]]))
