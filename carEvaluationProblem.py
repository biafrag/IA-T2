import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# classe singleton para que o data set nao seja criado todas as vezes
# Lê o csv com os dados e transforma em um data set, separando em dados de teste e de treino
class _CarEvaluation(object):
    
    _instance = None
    
    def __init__(self):
        self.seed = 7  # para manter todos os resultados reproduziveis
        np.random.seed(self.seed)

        # load dataset
        self.dataframe = pd.read_csv(r"car_evaluation.csv")

        # Assign names to Columns
        self.dataframe.columns = ['buying','maint','doors','persons','lug_boot','safety','classes']

        # Encode Data, transformando o valor dos atributos de strings para numeros
        self.dataframe.buying.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
        self.dataframe.maint.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
        self.dataframe.doors.replace(('2','3','4','5more'),(1,2,3,4), inplace=True)
        self.dataframe.persons.replace(('2','4','more'),(1,2,3), inplace=True)
        self.dataframe.lug_boot.replace(('small','med','big'),(1,2,3), inplace=True)
        self.dataframe.safety.replace(('low','med','high'),(1,2,3), inplace=True)
        self.dataframe.classes.replace(('unacc','acc','good','vgood'),(1,2,3,4), inplace=True)

        dataset = self.dataframe.values

        # x -> atributos do problema são 5, de 0 até 5
        X = dataset[:,0:6]
        # Y -> classificação de cada caso, coluna 6 do dataframe
        Y = np.asarray(dataset[:,6], dtype="S6")


        # Separando os dados para teste e os dados para treinamento do método usando 80% treino/20% teste,
        # os dados são separados de forma aleatória pela função "train_test_split"
        self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(X, Y, test_size=0.2)

    def printHead(self):
        print("dataframe.head: ", self.dataframe.head())
        print("dataframe.describe: ", self.dataframe.describe())

    def showDataHist(self):
        # Criando o Histograma com a distribuição das classes
        plt.hist((self.dataframe.classes))
        # Histograma de cada atributo do problema
        self.dataframe.hist()
        k = [3,5,7,9,11]
        y = [0.9050925925925926,0.9050925925925926,0.9606481481481481,0.9467592592592593,0.9259259259259259]
        plt.plot(k,y)
        plt.show()

# funçao que deve ser chamada para criar a classe,
# garante que a classe dos dados só é criada uma vez
def CarProblem():
    if _CarEvaluation._instance is None:
        _CarEvaluation._instance = _CarEvaluation()
    return _CarEvaluation._instance