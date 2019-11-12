import pandas as pd
import random

from datetime import datetime, timedelta
from json import dumps
from kafka import KafkaProducer
from time import sleep

# Kafka broker address and topic names definition
SERVER_URI = "<KAFKA BROKER URI>"
SERVER_PORT = "<KAFKA BROKER PORT>"
RAW_DATA_TOPIC = "raw_data"
LAB_DATA_TOPIC = "lab_data"
# Input data path
RAW_DATA_PATH = "prepared_real_data.csv"
LAB_DATA_PATH = "lab_data.csv"
# Synchronization lag and max delay between messages
DATA_LAG = 2 # Hours
MAX_DELAY = 8000 # Milliseconds

# Kafka producer params for connection and serializartion
producer = KafkaProducer(bootstrap_servers=f"{SERVER_URI}:{SERVER_PORT}",
                         key_serializer=lambda x: dumps(x).encode('utf-8'),
                         value_serializer=lambda x: dumps(x).encode('utf-8'))

# Read input data
raw_data_df = pd.read_csv(RAW_DATA_PATH, decimal='.')
lab_data_df = pd.read_csv(LAB_DATA_PATH, decimal='.')

# Iterate over both dataframes producing new data with random delays synchronized with the specified lag
last_hour = 0
counter = 0
for index, row in raw_data_df.iterrows():
    raw_data = row.to_dict()
    current_hour = raw_data['hour']
    # When hour changes send also the lab measure (after the specified lag)
    if last_hour != current_hour:
        if counter >= DATA_LAG:
            ix = counter - DATA_LAG
            row = lab_data_df.iloc[ix].to_dict()
            # Tag the lagged measure with the date to join
            date = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S') + timedelta(hours=DATA_LAG)
            key = date.strftime('%Y-%m-%d %H:%M:%S')
            value = {'date': key, 'silica_concentrate': row['silica_concentrate']}
        # For the first two hours send default values
        elif counter == 0:
            # When there hasn't been lab measures, default to this
            default_date = datetime(2017, 9, 1, 7)
            key = default_date.strftime('%Y-%m-%d %H:%M:%S')
            value = {'date': key, 'silica_concentrate': 2.0}
        elif counter == 1:
            # When there hasn't been lab measures, default to this
            default_date = datetime(2017, 9, 1, 8)
            key = default_date.strftime('%Y-%m-%d %H:%M:%S')
            value = {'date': key, 'silica_concentrate': 2.0}
        # Send lab data to the specified topic as key/value
        producer.send(LAB_DATA_TOPIC, key=key, value=value)
        last_hour = current_hour
        counter += 1
    
    # Send always every raw data measure as value-only
    producer.send(RAW_DATA_TOPIC, key=raw_data['date'], value=raw_data)

    # Sleep for up to MAX_DELAY ms
    sleep(random.uniform(0, MAX_DELAY) / 1000)