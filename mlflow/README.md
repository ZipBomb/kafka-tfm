# How to run MLflow on AWS EB for concurrent, safe and scalable experimentation

<div style="text-align: justify">
This directory contains all the configuration files needed to pack and deploy an MLflow tracking server as an Elastic Beanstalk application for better security and scalability. Up until now (v0.9), MLflow doesn't provide authentication mechanisms by itself. This allows everyone to see the results of the experimentations and even delete them. This solution takes advantage of EB to set up a nginx web server with authentication methods as a reverse proxy for external requests to the MLflow server. Using EB, the application can be also automatically scaled with more instances as users increase requests for recording experiments and getting predictions from the server. Metadatada containing the metrics stored for every experiment is stored locally on the containers while artifacts (serialized trained models and images) are stored in an S3 bucket.
</div>

## Getting Started

<div style="text-align: justify">
These instructions will get you an EB instance running with authentication and load-balancing capabilities for MLflow experimentation.
</div>

### Prerequisites

You will need the following things prior to the installation:

* An S3 bucket
* An IAM instance profile with write permissions on the bucket

### Deployment

1. Generate a valid authentication HTTP string with your credentials
   ```bash
   htpasswd -nb username password
   ```
2. Paste the output in line 7 of *.ebextensions/01-http_basic_auth_mlflow.config*
3. Zip the files
  ```bash
  zip -r mlflow-release.zip Dockerfile Dockerrun.aws.json .ebextensions
  ```
4. Create an AWS EB web server environment choosing Docker as platform and attach the zip from the last step.
5. In Configuration > Software create a new property called BUCKET with the name of your S3 bucket as the value.
6. In Configuration > Instances set the instance type as desired.
7. In Configuration > Capacity set the environment type as *Load balanced* and set the minimum and maximum instances as needed.
8. In Configuration > Security leave service role by default and set the IAM instance profile as the one with write permissions over your S3 bucket.
9. Deploy the application.
10. Access the URL assigned to your environment and you should be prompted for username/password.
11. Once the form is fulfilled you should get to MLflow's UI.
12. Profit ??? 🔥🔥🔥

### How to use

Now that MLflow is running with an authentication server in front you should do the following to record the experiments results and artifacts:

```bash
export MLFLOW_TRACKING_URI=<your_eb_uri_port_80>
export MLFLOW_TRACKING_USERNAME=<your_username>
export MLFLOW_TRACKING_PASSWORD=<your_password>
# If you are not using IAM roles set this also
export AWS_ACCESS_KEY_ID=<your_aws_access_key_id>
export AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key>
```
<div style="text-align: justify">
Every MLflow command you execute on your session will run against the remote tracking server so you can create experiments or serve models remotely. When you run experiments, via local code with MLflow calls or via MLprojects, the code will be run locally and metrics will be stored on the remote server so you can access them through the web. Every artifact generated as well as images will be stored on the selected S3 instance.
<div/>

### Based on

* [Medium](https://medium.com/@josemrivera/running-mlflow-on-aws-to-track-your-machine-learning-experiments-e192830f996c) - Running MLflow on AWS to track your machine learning experiments
* [Amazon Docs](https://docs.aws.amazon.com/codedeploy/latest/userguide/getting-started-create-iam-instance-profile.html#getting-started-create-iam-instance-profile-console) - Create an IAM Instance Profile for Your Amazon EC2 Instances
