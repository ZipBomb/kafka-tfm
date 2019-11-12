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
from sklearn.svm import SVR
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
    ax.errorbar(param_values, training_err_by_criteria, yerr = training_dev_by_criteria)
    ax.errorbar(param_values, test_err_list)
    plt.xlim(param_values[0]-1, param_values[len(param_values)-1]+1)
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
def rbf_svr_tuning(c = [0.001, 0.01, 0.1, 1, 10], gamma = [0.001, 0.01, 0.1, 1, 10], k = 5,
             train_data_path = '../data/training_data.csv', save_model = False, tracking_uri = "http://0.0.0.0:5000"):

    # Log the parameters with mlflow
    mlflow.log_param("c", c)
    mlflow.set_tag("k", k)

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Get data shuffled and split into training and test sets
    mdr = MiningDataReader(path = train_data_path)
    (variable_names, X_train, X_test, y_train, y_test) = mdr.get_splitted_data()

    pipeline = Pipeline(steps = [('scaling', StandardScaler()),
                                 ('regression', SVR(kernel = 'rbf'))])

    ### TRAINING ###
    ################

    # Generate grid search for hyperparam tuning
    hyperparams = {}
    hyperparams['regression__C'] = c
    hyperparams['regression__gamma'] = gamma

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
    best_c = modelCV.best_params_['regression__C']
    best_gamma = modelCV.best_params_['regression__gamma']

    print("\nTraining finished. Evaluating model...\n")

    ### EVALUATION ###
    ##################

    # Criteria is C
    criteria = 'c'
    mlflow.set_tag("criteria", criteria)
    param_values = c

    # Predict test data variying criteria param and evaluate the models
    training_err_by_criteria, training_dev_by_criteria, test_err_list = [], [], []
    rmse_score, mae_score, r2_score = -1, -1, -1
    feature_names, feature_importances = [], []
    for param_value in tqdm(param_values):
        model = Pipeline(steps = [('scaler', StandardScaler()),
                                  ('regression', SVR(
                                        C = param_value,
                                        gamma = best_gamma,
                                        kernel = 'rbf'))])
        param = {'regression__C': param_value, 'regression__gamma': best_gamma}

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
                                        ('regression', SVR(
                                            C = param_list[best_index]['regression__C'],
                                            gamma = best_gamma,
                                            kernel = 'rbf'))])

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
    values = []
    try:
        for value in range_string.replace("[", "").replace("]", "").split(","):
            values.append(float(value))
        return values
    except:
        raise argparse.ArgumentTypeError(f"takes one or more values in the form [c0,c1,c2...cN]")

# Custom types for input validation
def check_valid_c(value):
    try:
        # Option 1: Fixed value
        c = float(value)
        return [c]
    except ValueError:
        # Option 2: List of values
        list_values = parse_range_string(value)
        return list_values

if __name__ == "__main__":
    # Set the input argument parser
    parser = argparse.ArgumentParser(
        prog = 'rbf_svr.py',
        description = "Performs hyperparameter CV tuning for scikit-learn's Lasso implementation to solve "
                    + "Kaggle's 'Quality prediction in a mining process' problem and logs the experiment with mlflow.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-c', '--c', type = check_valid_c, metavar = 'N', default = [1],
                        help = 'list for C values to try')
    parser.add_argument('-g', '--gamma', type = check_valid_c, metavar = 'N', default = [1],
                            help = 'list for gamma values to try')
    parser.add_argument('-k', '--k_folds', metavar = 'K', choices = ['5', '10'], default = 5,
                        help = 'number of folds for CV')
    parser.add_argument('-i', metavar = 'PATH', default = '../../data/training_data.csv',
                        help = 'path to the training data CSV file')
    parser.add_argument('-ns', '--nosave', action = 'store_false',
                        help = 'tells mlflow not to log the final model and plots into an artifact')

    args = parser.parse_args()

    # Execute hyperparameter tuning
    rbf_svr_tuning(c = args.c,
                   gamma = args.gamma,
                   k = int(args.k_folds),
                   train_data_path = args.i,
                   save_model = args.nosave)
