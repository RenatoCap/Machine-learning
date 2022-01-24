from random import randrange
from csv import reader

class Utils: 
    
    def cargarDatos_csv(filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                dataset.append(row)
        return dataset
    

    def str_column_to_float(dataset, column):
        for row in dataset:
            try:
                row[column] = round(float(row[column].strip()), 3)
            except ValueError:
                row[column] = str(row[column].strip())
    

    def getColumn(dataset, column=0):
        column_value = list()
        for i in range(len(dataset)):
            column_value.append(dataset[i][column])
        return column_value
    

    def minmaxDatos(dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax
    

    def normalize_dataset(dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    

    def cross_validation_split(dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    
    def accuracy_metrics(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct +=1
        return correct / float(len(actual)) * 100.0
    
    
    