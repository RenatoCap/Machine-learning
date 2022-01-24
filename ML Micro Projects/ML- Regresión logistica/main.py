import matplotlib.pyplot as plt
import models
from utils import Utils
from random import seed


if __name__ == '__main__':
    LogisticRegresion = models.LogisticRegresion()
    Perceptron = models.Perceptron()
    DecisionTree = models.DecisionTree()
    seed(1)

    #Importación del dataset
    path = './in/dataset_definitive.csv'
    dataset = Utils.cargarDatos_csv(path)
    print('\n') 
    print('-'*30 + 'Primera 5 filas de mi Dataset' + 30*'-')
    print(dataset[:6])

    #Preprocesamiento
    for i in range(len(dataset[0])):
        Utils.str_column_to_float(dataset, i)
    print('\n')    
    print('-'*30 + 'Datos preprocesados' + 30*'-')
    print(dataset[:10])
    dataset_values = dataset[1:]

    minmax = Utils.minmaxDatos(dataset_values)
    Utils.normalize_dataset(dataset_values, minmax)

    #Gráfico de mis datos
    print('-'*30 + 'Grafico de nuestros datos' + 30*'-')
    plt.scatter(Utils.getColumn(dataset_values), Utils.getColumn(dataset_values,1), c=Utils.getColumn(dataset_values,2), cmap="Spectral")
    plt.show()

    #Utilización de modelos
    print('='*30 + 'Usando Regresión logistica' + '='*30)
    print('-'*30 + 'Entrenamiento' + 30*'-')
    scores = LogisticRegresion.evaluateAlgorithm(dataset_values,LogisticRegresion.logisticRegresion)
    
    print('\n')
    print('-'*30 + 'Precisión' + 30*'-')
    print('Scores: %s' %scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    print('\n')
    print('='*20 + 'Usando Perceptron' + '='*20)
    print('-'*30 + 'Entrenamiento' + 30*'-')
    scores = Perceptron.evaluateAlgorithm(dataset_values, Perceptron.perceptron)

    print('\n')
    print('-'*30 + 'Precisión' + 30*'-')
    print('Scores: %s' % scores)
    print('Mean Accuracy: %3.f%%' % (sum(scores)/float(len(scores))))

    print('\n')
    print('='*20 + 'Arbol de desición' + '='*20)
    scores = DecisionTree.evaluateAlgorithm(dataset_values, DecisionTree.decision_tree)

    print('\n')
    print('-'*30 + 'Precisión' + 30*'-')
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))




