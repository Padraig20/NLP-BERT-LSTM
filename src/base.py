from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from utils.dataloader import get_bbc_tokenized_ngrams
from utils.dataloader import get_spam_tokenized_ngrams
import sys

def print_results_test(y_test, y_pred, classifier_name):
    with open(output, 'a') as file:
        file.write("--------------" + classifier_name + "--------------\n\n\n")
        file.write(str(confusion_matrix(y_test, y_pred)))
        file.write("\n\n\n")
        file.write(str(classification_report(y_test, y_pred)))
        file.write("\n\n")
        file.write("--------------" + classifier_name + "--------------\n\n\n")


def print_results_opt(gs, classifier_name):
    results = gs.cv_results_
    i = gs.best_index_
    with open(output, 'a') as file:
        file.write("--------------" + classifier_name + "--------------\n")
        file.write("BEST SCORE: " + str(gs.best_score_) + "\n")
        file.write("BEST PARAMS: " + str(gs.best_params_) + "\n")
        file.write("BEST ESTIMATOR: " + str(gs.best_estimator_) + "\n")
        file.write("MEAN ACCURACY: " + str(results['mean_test_accuracy'][i]) + "\n")
        file.write("STANDARD ACCURACY: " + str(results['std_test_accuracy'][i]) + "\n")
        file.write("MEAN F1 SCORE: " + str(results['mean_test_f1_weighted'][i]) + "\n")
        file.write("STANDARD F1 SCORE: " + str(results['std_test_f1_weighted'][i]) + "\n")
        file.write("MEAN PRECISION: " + str(results['mean_test_precision_weighted'][i]) + "\n")
        file.write("STANDARD PRECISION: " + str(results['std_test_precision_weighted'][i]) + "\n")
        file.write("MEAN RECALL: " + str(results['mean_test_recall_weighted'][i]) + "\n")
        file.write("STANDARD RECALL: " + str(results['std_test_recall_weighted'][i]) + "\n")
        file.write("MEAN FITTING TIME: " + str(results['mean_fit_time'][i]) + "\n")
        file.write("MEAN SCORING TIME: " + str(results['mean_score_time'][i]) + "\n")
        file.write("--------------" + classifier_name + "--------------\n\n")


def gs_random_forest_classifier(X, y, scoring):
    param_grid = {
        "n_estimators": [10, 50, 100]
    }

    gs = GridSearchCV(estimator=RandomForestClassifier(),
                      param_grid=param_grid,
                      scoring=scoring,
                      refit='accuracy',
                      cv=5,
                      verbose=3)
    gs.fit(X, y)

    print_results_opt(gs, "RANDOM FOREST CLASSIFIER")


def gs_k_nearest_neighbors_classifier(X, y, scoring):
    param_grid = {
        "n_neighbors": [1, 3, 5, 10]
    }

    gs = GridSearchCV(estimator=KNeighborsClassifier(),
                      param_grid=param_grid,
                      scoring=scoring,
                      refit='accuracy',
                      cv=5,
                      verbose=5)
    gs.fit(X, y)

    print_results_opt(gs, "K NEAREST NEIGHBORS CLASSIFIER")


def gs_mlp_classifier(X, y, scoring):
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (150,)]
    }

    gs = GridSearchCV(estimator=MLPClassifier(),
                      param_grid=param_grid,
                      scoring=scoring,
                      refit='accuracy',
                      cv=5,
                      verbose=3)
    gs.fit(X, y)

    print_results_opt(gs, "MLP CLASSIFIER")


def gs_decision_tree_classifier(X, y, scoring):
    param_grid = {}

    gs = GridSearchCV(estimator=DecisionTreeClassifier(),
                      param_grid=param_grid,
                      scoring=scoring,
                      refit='accuracy',
                      cv=5,
                      verbose=3)
    gs.fit(X, y)

    print_results_opt(gs, "DECISION TREE CLASSIFIER")


def test_random_forest_classifier(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print_results_test(y_test, y_pred, "RANDOM FOREST CLASSIFIER")


def test_k_nearest_neighbor_classifier(X_train, y_train, X_test, y_test):
    classifier = KNeighborsClassifier(1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print_results_test(y_test, y_pred, "K NEAREST NEIGHBOR CLASSIFIER")


def test_mlp_classifier(X_train, y_train, X_test, y_test):
    classifier = MLPClassifier(150)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print_results_test(y_test, y_pred, "MLP CLASSIFIER")


def test_decision_tree_classifier(X_train, y_train, X_test, y_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print_results_test(y_test, y_pred, "DECISION TREE CLASSIFIER")


def test_dummy_classifier(X_train, y_train, X_test, y_test):
    classifier = DummyClassifier(strategy='uniform')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print_results_test(y_test, y_pred, "DUMMY CLASSIFIER")


def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test


import argparse

parser = argparse.ArgumentParser(description='Hyperparameter optimization and testing for base algorithms.')

parser.add_argument('mode', type=str, help='"opt" | "test"')
parser.add_argument('dataset', type=str, help='"bbc" | "spam"')
parser.add_argument('report', type=str, help='path/to/report.txt')

parser.add_argument('-a', '--augmentation', type=bool, default=False,
                    help='Choose whether data augmentation should be performed before training.')

# Parse the command-line arguments
args = parser.parse_args()

mode = args.mode
dataset = args.dataset
output = args.report
augmentation = args.augmentation

if mode == "opt":
    if dataset == "bbc": 
        df_x, df_y = get_bbc_tokenized_ngrams(True, 2, augmentation)
    elif dataset == "spam":
        df_x, df_y = get_spam_tokenized_ngrams(True, 2, augmentation)
    else:
        sys.exit(1)

    scoring = ['accuracy', 'recall_weighted', 'f1_weighted', 'precision_weighted']

    gs_mlp_classifier(df_x, df_y, scoring)
    gs_decision_tree_classifier(df_x, df_y, scoring)
    gs_random_forest_classifier(df_x, df_y, scoring)
    gs_k_nearest_neighbors_classifier(df_x, df_y, scoring)
elif mode == "test":
    if dataset == "bbc": 
        df_train_x, df_train_y, df_test_x, df_test_y = get_bbc_tokenized_ngrams(False, 2, augmentation)
    elif dataset == "spam":
        df_train_x, df_train_y, df_test_x, df_test_y = get_spam_tokenized_ngrams(False, 2, augmentation)
    else:
        sys.exit(1)

    test_mlp_classifier(df_train_x, df_train_y, df_test_x, df_test_y)
    test_decision_tree_classifier(df_train_x, df_train_y, df_test_x, df_test_y)
    test_random_forest_classifier(df_train_x, df_train_y, df_test_x, df_test_y)
    test_k_nearest_neighbor_classifier(df_train_x, df_train_y, df_test_x, df_test_y)
    test_dummy_classifier(df_train_x, df_train_y, df_test_x, df_test_y)
else:
    print("Unknown Mode!")
    print("Usage:\t\tpython base.py (\"opt\"|\"test\") (\"bbc\"|\"spam\") output.txt")
    sys.exit(1)
