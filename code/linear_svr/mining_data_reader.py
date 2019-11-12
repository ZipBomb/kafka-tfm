import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class MiningDataReader:
    '''
    Builds a pandas dataframe from Kaggle's "Quality prediction in a mining process"
    dataset after variable selection and other preprocessing steps.
    See https://www.kaggle.com/edumagalhaes/quality-prediction-in-a-mining-process
    '''
    def __init__(self, path='../data/training_data.csv', decimal_sep='.'):
        # Define categorical and numerical feature names
        self.categorical_features = ['day_of_month','day_of_week','hour']
        self.numeric_features = ['% Iron Feed','Starch Flow','Amina Flow','Ore Pulp Flow',
                                   'Ore Pulp pH','Ore Pulp Density','Flotation Column 01 Air Flow',
                                   'Flotation Column 02 Air Flow','Flotation Column 03 Air Flow',
                                   'Flotation Column 04 Air Flow','Flotation Column 05 Air Flow',
                                   'Flotation Column 06 Air Flow','Flotation Column 07 Air Flow',
                                   'Flotation Column 01 Level','Flotation Column 02 Level','Flotation Column 03 Level',
                                   'Flotation Column 04 Level','Flotation Column 05 Level','Flotation Column 06 Level',
                                   'Flotation Column 07 Level','sc_lag2','% Silica Concentrate']
        self.features = self.categorical_features + self.numeric_features
        # Read data into a pandas dataframe
        self.df = self.preprocess_data(pd.read_csv(path, decimal=decimal_sep))

    '''Executes preprocessing steps for the original pandas dataframe'''
    def preprocess_data(self, df):
        # # STEP 1: Extract date information
        # df['day_of_month'] = df['date'].dt.day
        # df['day_of_week'] = df['date'].dt.dayofweek
        # df['hour'] = df['date'].dt.hour
        # # STEP 2: Group data with mean
        # df = df.groupby(['date']).mean()
        # # STEP 3: Drop correlated columns
        # df.drop(columns=['% Silica Feed','% Iron Concentrate'], inplace=True)
        # # STEP 4: Reset index and drop raw date column
        # df = df.reset_index()
        # df.drop(columns=['date'], inplace=True)
        # STEP 5: Reorder columns
        df = df[self.features]
        
        return df

    '''Returns the loaded pandas dataframe'''
    def get_data_as_pandas(self):
        return self.df

    '''Returns the original pandas dataframe converted into a numpy matrix and splitted into X and y'''
    def get_data_as_numpy(self):
        # Split features from the target attribute
        df = self.df.values
        X = df[:, 0:df.shape[1]-1]
        y = df[:, df.shape[1]-1]
        return (X, y)

    '''Splits data into training and test sets with random shuffled split'''
    def get_splitted_data(self, test_size=0.2, rand_seed=42):
        (X, y) = self.get_data_as_numpy()
        # Random seed on numpy for reproducibility
        np.random.seed(rand_seed)
        # Shuffle and divide the observations in training and test sets
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=rand_seed)
        # Get column names
        variable_names = (self.features[:-1])

        return (variable_names, X_train, X_test, y_train, y_test)
