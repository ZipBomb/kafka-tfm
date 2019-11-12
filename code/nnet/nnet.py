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
from sklearn.neural_network import MLPRegressor
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
    fig, ax = plt.subplots(figsize=(15, 10))
    # fix for points with more than one item
    ticks = list(map(lambda param: str(param), param_values))
    ax.errorbar(ticks, training_err_by_criteria, yerr = training_dev_by_criteria)
    ax.errorbar(ticks, test_err_list)
    # plt.xlim(param_values[0]-1, param_values[len(param_values)-1]+1)
    plt.legend(['Training', 'Test'], loc = 'upper right')
    plt.xlabel(criteria)
    plt.ylabel('MSE')
    plt.savefig('./img/evaluation.png', bbox_inches = 'tight')

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

# Performs hyperparameter tuning over Lasso scikit-learn implementation, logs metrics and stores results
def nnet_tuning(n_layers = [1, 2, 1], layer_size = [20, 21, 1], k = 5,
                train_data_path = '../data/training_data.csv', save_model = False, tracking_uri = "http://0.0.0.0:5000"):

    # Log the parameters with mlflow
    mlflow.log_param("n_layers", n_layers)
    mlflow.set_tag("layer_size", layer_size)

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Get data shuffled and split into training and test sets
    mdr = MiningDataReader(path = train_data_path)
    (variable_names, X_train, X_test, y_train, y_test) = mdr.get_splitted_data()

    pipeline = Pipeline(steps = [('scaling', StandardScaler()),
                                 ('regression', MLPRegressor(random_state = RANDOM_SEED))])

    ### TRAINING ###
    ################

    # Generate all combinations for number of layers and layer size
    neurons_per_layer = tuple(np.arange(layer_size[0], layer_size[1], layer_size[2]))
    hls_values = []
    for layers_num in np.arange(n_layers[0], n_layers[1], n_layers[2]):
        hls_values.append([x for x in itertools.product(neurons_per_layer,repeat=layers_num)])

    # Flatten the list
    hls_values = [item for sublist in hls_values for item in sublist]

    # Generate grid search for hyperparam tuning
    hyperparams = {}
    hyperparams['regression__hidden_layer_sizes'] = hls_values

    print("Training started...\n")

    # Create an instance of Random Forest Regressor and fit the data for the grid parameters using all processors
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

    print(f"\nBest parameter set found for the training set:\n{modelCV.best_params_}")

    # Store the index of the best combination
    best_index = param_list.index(modelCV.best_params_)

    # Get the best values for hyperparams
    best_hls = modelCV.best_params_['regression__hidden_layer_sizes']

    print("\nTraining finished. Evaluating model...\n")

    ### EVALUATION ###
    ##################

    # Criteria is hidden_layer_sizes
    criteria = 'hidden_layer_sizes'
    mlflow.set_tag("criteria", criteria)
    param_values = hls_values

    # Predict test data variying criteria param and evaluate the models
    training_err_by_criteria, training_dev_by_criteria, test_err_list = [], [], []
    rmse_score, mae_score, r2_score = -1, -1, -1
    feature_names, feature_importances = [], []
    for param_value in tqdm(param_values):
        model = Pipeline(steps = [('scaler', StandardScaler()),
                                  ('regression', MLPRegressor(
                                        hidden_layer_sizes = param_value,
                                        random_state = RANDOM_SEED))])
        param = {'regression__hidden_layer_sizes': param_value}

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
                                        ('regression', MLPRegressor(
                                            hidden_layer_sizes = param_list[best_index]['regression__hidden_layer_sizes'])
                                        )])

        # Train the best model with all the data (training + test)
        full_X = np.vstack((X_train, X_test))
        full_y = np.concatenate((y_train, y_test))
        final_model.fit(full_X, full_y)

        # Log plots and model with mlflow
        mlflow.log_artifacts('./img')
        mlflow.sklearn.log_model(final_model, 'model')

    # Log results with mlflow
    mlflow.log_metric("train_mse", training_err_list[best_index])
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

def check_valid_split(value):
    try:
        # Option 1: Fixed value
        min_split = int(value)
        if min_split < 1:
            raise argparse.ArgumentTypeError(f"takes one integer >=  or a range in the form: <min,max,step>")
        else:
            return [min_split, min_split + 1, 1]
    except ValueError:
        # Option 2: Range of values
        range_values = parse_range_string(value)
        if range_values[0] < 1:
            raise argparse.ArgumentTypeError(f"takes one integer >=  or a range in the form: <min,max,step>")
        return range_values

if __name__ == "__main__":
    # Set the input argument parser
    parser = argparse.ArgumentParser(
        prog = 'nnet.py',
        description = "Performs hyperparameter CV tuning for scikit-learn's Lasso implementation to solve "
                    + "Kaggle's 'Quality prediction in a mining process' problem and logs the experiment with mlflow.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-nl', '--n_layers', type = check_valid_split, metavar = 'N', default = [1],
                        help = 'list for the number of layers to try')
    parser.add_argument('-ls', '--layer_size', type = check_valid_split, metavar = 'N', default = [1],
                            help = 'list for the number of neurons per layer to try')
    parser.add_argument('-k', '--k_folds', metavar = 'K', choices = ['5', '10'], default = 5,
                        help = 'number of folds for CV')
    parser.add_argument('-i', metavar = 'PATH', default = '../../data/training_data.csv',
                        help = 'path to the training data CSV file')
    parser.add_argument('-ns', '--nosave', action = 'store_false',
                        help = 'tells mlflow not to log the final model and plots into an artifact')

    args = parser.parse_args()

    # Execute hyperparameter tuning
    nnet_tuning(n_layers = args.n_layers,
                layer_size = args.layer_size,
                k = int(args.k_folds),
                train_data_path = args.i,
                save_model = args.nosave)
