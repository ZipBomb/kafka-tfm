# -*- coding: utf-8 -*-
import argparse
import errno
import itertools
import math
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import os
import pydot
import random
import scipy.stats as stats
import shutil
import sklearn.metrics as metrics
import sys

from dask.diagnostics import ProgressBar
from dask_ml.model_selection import GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mining_data_reader import MiningDataReader

# Global constants
RANDOM_SEED = 42

#######################################################################################################################
#######################################################################################################################

# Function to clear output images folder
def empty_img_folder(path = './img'):
    if not os.path.exists(path):
        os.makedirs(path)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

# Function to plot training vs test error
def plot_errors(criteria, param_values, training_err_by_criteria, training_dev_by_criteria, test_err_list):
    # Fix for when max_depth is specified as None
    if criteria == 'max_depth' and param_values[0] is None:
        return
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.errorbar(param_values, training_err_by_criteria, yerr = training_dev_by_criteria)
    ax.errorbar(param_values, test_err_list)
    plt.xlim(param_values[0]-1, param_values[len(param_values)-1]+1)
    plt.legend(['Training', 'Test'], loc = 'upper right')
    plt.xlabel(criteria)
    plt.ylabel('MSE')
    plt.savefig('./img/evaluation.png', bbox_inches = 'tight')

# Function to plot feature importances
def plot_feature_importances(importances, names):
    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.barh(y_pos, importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    plt.savefig('./img/importances.png', bbox_inches = 'tight')

# Compute metrics from predicted output and y_test
def get_test_metrics(training_err, y_test, prediction):
    training_mse = training_err
    test_mse = metrics.mean_squared_error(y_test, prediction)
    rmse = math.sqrt(test_mse)
    mae = metrics.mean_absolute_error(y_test, prediction)
    r2 = metrics.r2_score(y_test, prediction)
    return (training_mse, test_mse, rmse, mae, r2)

#######################################################################################################################
#######################################################################################################################

# Performs hyperparameter tuning over CART scikit-learn implementation, logs metrics and stores the results
def cart_tuning(max_depth = None, min_samples_leaf = [1, 2, 1], min_samples_split = [2, 3, 1], k = 5,
                train_data_path = '../data/training_data.csv', save_model = False, tracking_uri = "http://0.0.0.0:5000"):

    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('min_samples_leaf', min_samples_leaf)
    mlflow.log_param('min_samples_split', min_samples_split)
    mlflow.set_tag("k", k)

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Get data shuffled and split into training and test sets
    mdr = MiningDataReader(path = train_data_path)
    (variable_names, X_train, X_test, y_train, y_test) = mdr.get_splitted_data()

    pipeline = Pipeline(steps = [('scaler', StandardScaler()),
                                 ('regression', DecisionTreeRegressor(random_state = RANDOM_SEED))])

    ### TRAINING ###
    ################

    # Generate grid search for hyperparam tuning
    hyperparams = {}
    hyperparams['regression__max_depth'] = [None] if max_depth is None else np.arange(
                                                     max_depth[0], max_depth[1], max_depth[2]
                                                  )
    hyperparams['regression__min_samples_leaf'] = np.arange(
                                                     min_samples_leaf[0], min_samples_leaf[1], min_samples_leaf[2]
                                                  )
    hyperparams['regression__min_samples_split'] = np.arange(
                                                     min_samples_split[0], min_samples_split[1], min_samples_split[2]
                                                   )

    print("Training started...\n")

    # Create an instance of Decision Tree Regressor and fit the data for the grid parameters using all processors
    modelCV = GridSearchCV(estimator = pipeline,
                           param_grid = hyperparams,
                           cv = k,
                           scoring = 'neg_mean_squared_error',
                           n_jobs = -1)

    with ProgressBar():
        modelCV.fit(X_train, y_train)

    # Iterate over the results storing training error for each hyperparameter combination
    results = modelCV.cv_results_
    param_list, training_err_list, training_dev_list = [], [], []
    for i in range(len(results['params'])):
        param = results['params'][i]
        score = (-1) * results['mean_test_score'][i] # NEGATIVE MSE
        std = results['std_test_score'][i]
        param_list.append(param)
        training_err_list.append(score)
        training_dev_list.append(std)

    print(f"\nBest parameters set found for the training set:\n{modelCV.best_params_}")

    # Store the index of the best combination
    best_index = param_list.index(modelCV.best_params_)

    # Get the best values for hyperparams
    best_depth = modelCV.best_params_['regression__max_depth']
    best_samples_leaf = modelCV.best_params_['regression__min_samples_leaf']
    best_samples_split = modelCV.best_params_['regression__min_samples_split']

    print("\nTraining finished. Evaluating model...\n")

    ### EVALUATION ###
    ##################

    # Select the hyperparam with most values as the criteria for the study and calculate test error with the best value
    # obtained for the other hyperparameters so the individual effect of this parameter can be studied
    criteria = [('max_depth', len(hyperparams['regression__max_depth'])),
                ('min_samples_leaf', len(hyperparams['regression__min_samples_leaf'])),
                ('min_samples_split', len(hyperparams['regression__min_samples_split']))]
    criteria = sorted(criteria, key = lambda x: x[1], reverse = True)[0][0]

    mlflow.set_tag("criteria", criteria)

    param_values = []
    if criteria == 'max_depth':
        if max_depth is None:
            param_values = [None]
        else:
            param_values = range(max_depth[0], max_depth[1], max_depth[2])
    elif criteria == 'min_samples_leaf':
        param_values = range(min_samples_leaf[0], min_samples_leaf[1], min_samples_leaf[2])
    else:
        param_values = range(min_samples_split[0], min_samples_split[1], min_samples_split[2])

    # Predict test data variying criteria param and evaluate the models
    training_err_by_criteria, training_dev_by_criteria, test_err_list = [], [], []
    rmse_score, mae_score, r2_score = -1, -1, -1
    feature_names, feature_importances = [], []
    for param_value in tqdm(param_values):
        if criteria == 'max_depth':
            model = Pipeline(steps = [('scaler', StandardScaler()),
                                      ('regression', DecisionTreeRegressor(
                                            max_depth = param_value,
                                            min_samples_leaf = best_samples_leaf,
                                            min_samples_split = best_samples_split, random_state = RANDOM_SEED))])
            param = {'regression__max_depth': param_value, 'regression__min_samples_leaf': best_samples_leaf,
                     'regression__min_samples_split': best_samples_split}
        elif criteria == 'min_samples_leaf':
            model = Pipeline(steps = [('scaler', StandardScaler()),
                                      ('regression', DecisionTreeRegressor(
                                            max_depth = best_depth,
                                            min_samples_leaf = param_value,
                                            min_samples_split = best_samples_split, random_state = RANDOM_SEED))])
            param = {'regression__max_depth': best_depth, 'regression__min_samples_leaf': param_value,
                     'regression__min_samples_split': best_samples_split}
        else:
            model = Pipeline(steps = [('scaler', StandardScaler()),
                                      ('regression', DecisionTreeRegressor(
                                            max_depth = best_depth,
                                            min_samples_leaf = best_samples_leaf,
                                            min_samples_split = param_value, random_state = RANDOM_SEED))])
            param = {'regression__max_depth': best_depth, 'regression__min_samples_leaf': best_samples_leaf,
                     'regression__min_samples_split': param_value}

        # Fit model and evaluate results
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        index = param_list.index(param)
        training_err = training_err_list[index]
        training_dev = training_dev_list[index]
        (training_mse, test_mse, rmse, mae, r2) = get_test_metrics(training_err, y_test, prediction)
        # Store metrics
        training_err_by_criteria.append(training_mse)
        training_dev_by_criteria.append(training_dev)
        test_err_list.append(test_mse)
        # Set aditional metrics for the best combination
        if index == best_index:
            rmse_score = rmse
            mae_score = mae
            r2_score = r2

    # Generate the plots
    empty_img_folder()
    plot_errors(criteria, param_values, training_err_by_criteria, training_dev_by_criteria, test_err_list)

    # Once hyperparameters are selected, train and save the best model
    if save_model:
        print("\nEvaluation finished. Training final model with train + test data with the best hyperparameters...")
        final_model = Pipeline(steps = [('scaler', StandardScaler()),
                                        ('regression', DecisionTreeRegressor(
                                            max_depth = param_list[best_index]['regression__max_depth'],
                                            min_samples_leaf = param_list[best_index]['regression__min_samples_leaf'],
                                            min_samples_split = param_list[best_index]['regression__min_samples_split']))])

        # Train the best model with all the data (training + test)
        full_X = np.vstack((X_train, X_test))
        full_y = np.concatenate((y_train, y_test))
        final_model.fit(full_X, full_y)

        # Get a barplot with feature importances
        feature_importances = final_model.named_steps['regression'].feature_importances_
        plot_feature_importances(feature_importances, variable_names)

        # Create a visual representation of the tree and convert it to PNG
        tree_graph = tree.export_graphviz(final_model.named_steps['regression'],
                                          out_file = '/tmp/tree.dot',
                                          max_depth = 4)
        (graph,) = pydot.graph_from_dot_file('/tmp/tree.dot')
        graph.write_png('./img/tree.png')

        # Log plots and model with mlflow
        mlflow.log_artifacts('./img')
        mlflow.sklearn.log_model(final_model, 'model')

    # Log results with mlflow
    mlflow.log_metric("training_mse", training_err_list[best_index])
    mlflow.log_metric("test_mse", min(test_err_list))
    mlflow.log_metric("rmse", rmse_score)
    mlflow.log_metric("mae", mae_score)
    mlflow.log_metric("r2", r2_score)
    mlflow.set_tag("best_params", param_list[best_index])

    # Output the results
    print(f'''
-----------------------------------------------------------------------------------------------------------------------
RESULTS
-----------------------------------------------------------------------------------------------------------------------
Best params: {param_list[best_index]}
Training MSE: {training_err_list[best_index]}
Test MSE: {min(test_err_list)}
RMSE: {rmse_score}
MAE: {mae_score}
R2: {r2_score}
-----------------------------------------------------------------------------------------------------------------------
''')

#######################################################################################################################
#######################################################################################################################

# Checks if range values for hyperparams are specified in the correct way
def parse_range_string(range_string):
    try:
        range_string.replace(" ", "")
        (min_val, max_val, step) = range_string.split(",")
        return [int(min_val), int(max_val), int(step)]
    except:
        raise argparse.ArgumentTypeError(f"takes one positive integer or a range in the form: <min,max,step>")

# Custom types for input validation
def check_valid_depth(value):
    if value == "None":
        return None
    try:
        # Option 1: Fixed value
        max_depth = int(value)
        if max_depth < 1:
            raise argparse.ArgumentTypeError(f"takes one positive integer or a range in the form: <min,max,step>")
        else:
            return [max_depth, max_depth + 1, 1]
    except ValueError:
        # Option 3: Range of values
        range_values = parse_range_string(value)
        return range_values

def check_valid_leaf(value):
    try:
        # Option 1: Fixed value
        min_leaf = int(value)
        if min_leaf < 1:
            raise argparse.ArgumentTypeError(f"takes one integer >= 1 or a range in the form: <min,max,step>")
        else:
            return [min_leaf, min_leaf + 1, 1]
    except ValueError:
        # Option 2: Range of values
        range_values = parse_range_string(value)
        return range_values

def check_valid_split(value):
    try:
        # Option 1: Fixed value
        min_split = int(value)
        if min_split < 2:
            raise argparse.ArgumentTypeError(f"takes one integer >= 2 or a range in the form: <min,max,step>")
        else:
            return [min_split, min_split + 1, 1]
    except ValueError:
        # Option 2: Range of values
        range_values = parse_range_string(value)
        if range_values[0] < 2:
            raise argparse.ArgumentTypeError(f"takes one integer >= 2 or a range in the form: <min,max,step>")
        return range_values


if __name__ == "__main__":
    # Set the input argument parser
    parser = argparse.ArgumentParser(
        prog = 'cart_train.py',
        description = "Performs hyperparameter CV tuning for scikit-learn's CART implementation to solve Kaggle's "
                    + "'Quality prediction in a mining process' problem and logs the experiment with mlflow.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-md', '--max_depth', type = check_valid_depth, metavar = 'MD', default = "None",
                        help = 'value or range of values for the maximum depth of the tree, '
                            +  'if None runs until all leaves are pure')
    parser.add_argument('-msl', '--min_samples_leaf', type = check_valid_leaf, metavar = 'MSL', default = [1, 2, 1],
                        help = 'range for the minimum number of samples required to be at a leaf node (min >= 1)')
    parser.add_argument('-mss', '--min_samples_split', type = check_valid_split, metavar = 'MSS', default = [2, 3, 1],
                        help = 'value or range of values for the minimum number of samples '
                            +  'required to split an internal node (min >= 2)')
    parser.add_argument('-ts', '--test_size', metavar = 'TS', default = 0.15,
                        help = 'proportion of data saved for testing purposes')
    parser.add_argument('-k', '--k_folds', metavar = 'K', choices = ['5', '10'], default = 5,
                        help = 'number of folds for CV')
    parser.add_argument('-i', metavar = 'PATH', default = '../../data/training_data.csv',
                        help = 'path to the training data CSV file')
    parser.add_argument('-ns', '--nosave', action = 'store_false',
                        help = 'tells mlflow not to log the final model and plots into an artifact')
    parser.add_argument('-uri', '--tracking_uri', metavar = 'URI', default = "http://0.0.0.0:5000",
                        help = 'sets the uri for the mlflow tracking server where results will be stored')

    args = parser.parse_args()

    # Execute hyperparameter tuning
    cart_tuning(max_depth = args.max_depth,
                min_samples_leaf = args.min_samples_leaf,
                min_samples_split = args.min_samples_split,
                k = int(args.k_folds),
                train_data_path = args.i,
                save_model = args.nosave,
                tracking_uri = args.tracking_uri)
