# How to train models with our custom MLflow Projects wrappers for regression

<div style="text-align: justify">

</div>

## Getting Started

<div style="text-align: justify">
Once you have a remote MLflow server correctly configured and running (see [How to run MLflow on AWS EB for concurrent and scalable experimentation](../mlflow/README.md)) you are ready to start taking advantage of the scikit-learn/xgboost + MLflow tracking wrappers in /code.
</div>

### Prerequisites

You will need the following:

* An MLflow server instance running on a remote machine
* MLflow tracking related environment variables correctly set (at least MLFLOW_TRACKING_URI if your server runs with no authentication methods)
* Conda (https://docs.conda.io/en/latest/miniconda.html) with Python 3.7.3+
* MLflow 0.9 (conda install -c conda-forge mlflow=0.9.1)
* Graphviz (sudo apt install graphviz)

### How to use the CLI

- Managing experiments:

```bash
mlflow experiments <action>
```
```
Available actions are:
create   Create an experiment in the configured tracking server.
delete   Mark an experiment for deletion.
list     List all experiments in the configured tracking server.
rename   Renames an active experiment.
restore  Restore a deleted experiment.
```

- Running experiments (MLflow Projects):

```bash
mlflow run --experiment-id <experiment_id> -P <key_value_params> <method>
```
```
Some available methods are:
cart   Based on scikit-learn's DecissionTreeRegressor.
random_forest   Based on scikit-learn's RandomForestRegressor.
```
```
Some customizable parameters for the currently implemented algorithms are:
cart -> min_samples_split, min_samples_leaf, max_depth
random_forest ->  n_estimators

Params are specified as:
<param_name>=<value>
  where <value> can be a single digit, a range in the form <min,max,step> or a fixed list of values in the form [x0,x1,...,xN]
```

e.g. in the first two lines we create an experiment named CART and we train a regression model with the default dataset, specifying a range of values to test for the hyperparameter 'min_samples_split' and leaving the rest with default values. The following lines are examples of some methods.
```bash
mlflow experiments create CART # Returns ID = 0
mlflow run --experiment-id 0 -P min_samples_split=2,301,1 cart

mlflow experiments create RBF_SVR # Returns ID = 4
mlflow run --experiment-id 4 -P c=[1.e-02,1.e-01,1.e+00,\
1.e+01,1.e+02,1.e+03,1.e+04,1.e+05,1.e+06,1.e+07,1.e+08,\
1.e+09,1.e+10] -P gamma=[1.e-09,1.e-08,1.e-07,1.e-06,\
1.e-05,1.e-04,1.e-03,1.e-02,1.e-01,1.e+00,1.e+01,1.e+02,\
1.e+03] rbf_svr
```
When the run ends, parameters, performance metrics and the resulting model and artifacts should be accesible through the MLflow server instance.

- Serving experiments on a local REST API (MLflow Models):

```bash
mlflow pyfunc serve -r <run_id> -m model
```
where <run_id> is the ID of the run that produced the model that you want to use

### References

* [MLflow](https://www.mlflow.org/docs/latest/index.html) - MLflow Official Documentation
* [Databricks Forums](https://forums.databricks.com/questions/14693/how-can-spark-pipeline-models-in-mlflow-deployed-o.html) - How to deploy MLflow models in a local REST server
* [GitHub](https://github.com/mlflow/mlflow/tree/master/examples/hyperparam) - MLflow: Hyperparameter Tuning Example
* [TowardsDataScience](https://towardsdatascience.com/collaborate-on-mlflow-experiments-in-neptune-fb4f8f84a995) How to make your MLflow projects easy to share and collaborate on
