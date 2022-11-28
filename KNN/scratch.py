# %%
from random import randrange, seed
from csv import reader
from math import sqrt


def load_csv(filename):
    with open(filename, "r") as file:
        csv_reader = reader(file)
        dataset = [row for row in csv_reader if row]
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])


def str_column_to_int(dataset, column):
    class_values = set((row[column] for row in dataset))
    look_up = dict(zip(class_values, range(len(class_values))))
    for row in dataset:
        row[column] = look_up[row[column]]
    return look_up


def euclidean_distance(vec1, vec2):
    """Calculate the euclidean distance"""
    return sqrt(sum((i - j) ** 2 for i, j in zip(vec1[:-1], vec2[:-1])))


def get_nearest_neighbour(train, test_row, num_neighbors):
    distance = [(train_row, euclidean_distance(train_row, test_row))
                for train_row in train]
    distance.sort(key=lambda x: x[1])
    return [distance[i][0] for i in range(num_neighbors)]


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_nearest_neighbour(train, test_row, num_neighbors)
    output_value = [row[-1] for row in neighbors]
    pred = max(set(output_value), key=output_value.count)
    return pred


def train_test_split(dataset, test_size=0.60):
    dataset_copy = list(dataset)
    train_size = int(test_size * len(dataset))
    train = [dataset_copy.pop(random.randrange(len(dataset_copy)))
             for _ in range(train_size)]
    return train,


def cross_validation_split(dataset, folds=3):
    dataset_copy = list(dataset)
    folds_size = int(len(dataset_copy) / folds)
    return [[dataset_copy.pop(randrange(len(dataset_copy)))
             for _ in range(folds_size)]
            for _ in range(folds)]


def accuracy_metric(actual, predicted):
    correct = sum(actual[i] == predicted[i] for i in range(len(actual)))
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(algorithm, dataset, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        pred = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, pred)
        scores.append(accuracy)
    return scores


def k_nearest_algorithm(train, test, k):
    return [predict_classification(train, row, k)
            for row in test]


seed(1)
filename = "KNN/data/iris.csv"
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# evaluate algorithm
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(k_nearest_algorithm, dataset, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
