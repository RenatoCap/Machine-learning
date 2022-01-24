import matplotlib.pyplot as plt
from utils import Utils
from math import exp

class LogisticRegresion:
    
    def __init__(self):
        self.params = {
            'n_folds' : 5,
            'l_rate' : 0.1,
            'n_epoch' : 5
        }

    def prediction(row, coefficients):
        yhat = coefficients[0]
        for i in range(len(row) - 1):
            yhat += coefficients[i+1] * row[i]
        return 1.0 / (1.0 + exp(-yhat))


    def sgd_coefficients(self, train):
        coef = [0.0 for i in range(len(train))]
        for _ in range(self.params['n_epoch']):
            for row in train:
                yhat = LogisticRegresion.prediction(row, coef)
                error  = row[-1] - yhat
                coef[0] = coef[0] + self.params['l_rate'] * error * yhat * (1.0 - yhat)
                for i in range(len(row) - 1):
                    coef[i+1] = coef[i+1] + self.params['l_rate'] * error * yhat * (1.0 - yhat) * row[i]
        return coef

    
    def logisticRegresion(self, train, test):
        predictions = list()
        coef = LogisticRegresion.sgd_coefficients(self, train)
        for row in test:
            yhat = LogisticRegresion.prediction(row, coef)
            yhat = round(yhat)
            predictions.append(yhat)
        return (predictions)


    def evaluateAlgorithm(self, dataset, algorithm):
        folds = Utils.cross_validation_split(dataset, self.params['n_folds'])
        scores = list()
        count = 1
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
            predicted = algorithm(train_set, test_set)
            actual = [row[-1] for row in fold]
            print('='*20 + 'Fold Nº'+ str(count) + '='*20)
            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.suptitle("Predicted Vs Actual")
            ax1.scatter(Utils.getColumn(test_set), Utils.getColumn(test_set, 1), c=predicted, cmap = "Spectral")
            ax2.scatter(Utils.getColumn(test_set), Utils.getColumn(test_set, 1), c=actual, cmap = "seismic")
            plt.show()
            accuracy = Utils.accuracy_metrics(actual, predicted)
            scores.append(accuracy)
            count+=1
        return scores


class Perceptron:

    def __init__(self):
        self.params = {
            'n_folds' : 5,
            'l_rate' : 0.1,
            'n_epoch' : 500
        }
    
    def prediction(row, weights):
        activation = weights[0]
        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0


    def train_weights(self, train):
        weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(self.params['n_epoch']):
            sum_error = 0.0
            for row in train:
                prediction = Perceptron.prediction(row, weights)
                error = row[-1] - prediction
                sum_error += error**2
                weights[0] = weights[0] + self.params['l_rate'] * error
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + self.params['l_rate'] * error * row[i]
        return weights
    

    def perceptron(self, train, test):
        predictions = list()
        weights = Perceptron.train_weights(self, train)
        for row in test:
            prediction = Perceptron.prediction(row, weights)
            predictions.append(prediction)
        return(predictions)


    def evaluateAlgorithm(self, dataset, algorithm):
        folds = Utils.cross_validation_split(dataset, self.params['n_folds'])
        scores = list()
        count = 1
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
            predicted = algorithm(train_set, test_set)
            actual = [row[-1] for row in fold]
            print('='*20 + 'Fold Nº'+ str(count) + '='*20)
            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.suptitle("Predicted Vs Actual")
            ax1.scatter(Utils.getColumn(test_set), Utils.getColumn(test_set, 1), c=predicted, cmap = "Spectral")
            ax2.scatter(Utils.getColumn(test_set), Utils.getColumn(test_set, 1), c=actual, cmap = "seismic")
            plt.show()
            accuracy = Utils.accuracy_metrics(actual, predicted)
            scores.append(accuracy)
            count+=1
        return scores


class DecisionTree:

    def __init__(self):
        self.params = {
            'n_folds' : 5,
            'max_depth' : 5,
            'min_size' : 10
        }


    def predict_tree(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return DecisionTree.predict_tree(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return DecisionTree.predict_tree(node['right'], row)
            else:
                return node['right']


    def test_split(index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    
    def gini_index(groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    
    def get_split(dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = DecisionTree.test_split(index, row[index], dataset)
                gini = DecisionTree.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}


    def to_terminal(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
    

    def split(node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = DecisionTree.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = DecisionTree.to_terminal(left), DecisionTree.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = DecisionTree.to_terminal(left)
        else:
            node['left'] = DecisionTree.get_split(left)
            DecisionTree.split(node['left'], max_depth, min_size, depth+1)
        # process right child
        if len(right) <= min_size:
            node['right'] = DecisionTree.to_terminal(right)
        else:
            node['right'] = DecisionTree.get_split(right)
            DecisionTree.split(node['right'], max_depth, min_size, depth+1)


    def build_tree(train, max_depth, min_size):
        root = DecisionTree.get_split(train)
        DecisionTree.split(root, max_depth, min_size, 1)
        return root


    def decision_tree(self, train, test):
        tree = DecisionTree.build_tree(train, self.params['max_depth'], self.params['min_size'])
        predictions = list()
        for row in test:
            prediction = DecisionTree.predict_tree(tree, row)
            predictions.append(prediction)
        return(predictions)


    def evaluateAlgorithm(self, dataset, algorithm):
        folds = Utils.cross_validation_split(dataset, self.params['n_folds'])
        scores = list()
        count = 1
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
            predicted = algorithm(train_set, test_set)
            actual = [row[-1] for row in fold]
            print('='*20 + 'Fold Nº'+ str(count) + '='*20)
            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.suptitle("Predicted Vs Actual")
            ax1.scatter(Utils.getColumn(test_set), Utils.getColumn(test_set, 1), c=predicted, cmap = "Spectral")
            ax2.scatter(Utils.getColumn(test_set), Utils.getColumn(test_set, 1), c=actual, cmap = "seismic")
            plt.show()
            accuracy = Utils.accuracy_metrics(actual, predicted)
            scores.append(accuracy)
            count+=1
        return scores



