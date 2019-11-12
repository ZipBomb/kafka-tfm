from graphitesend import GraphiteClient
from json import loads
from kafka import KafkaConsumer

# Graphite host
GRAPHITE_HOSTNAME = "graphite"

# Kafka broker address and topic names definition
SERVER_URI = "<KAFKA_SERVER_URI>"
SERVER_PORT = "9092"
PREDICTION_TOPIC = "RAW_AND_PREDICTED"

# Kafka consumer setup
consumer = KafkaConsumer(f"{PREDICTION_TOPIC}",
                        bootstrap_servers=f"{SERVER_URI}:{SERVER_PORT}",
                        value_deserializer=lambda m: loads(m.decode('utf-8')))
# Graphite client setup
client = GraphiteClient(GRAPHITE_HOSTNAME)

# For every new message, send data to Graphite
for message in consumer:
    key = message.key.decode('utf-8')
    payload = message.value

    client.send("predicted", float(payload['PREDICTED']))
    client.send("real", float(payload['SILICA_CONCENTRATE']))