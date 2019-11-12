# Real-time visualization with Graphite + Grafana

<div style="text-align: justify">
Code to deploy a Docker Swarm containing Graphite and Grafana instances as well as a Python script to publish information consumed from a Kafka topic into Graphite. This allows us to plot real-time plant measurements alongside the predicted values given by our model into a Grafana dashboard. 
</div>

## Swarm deployment
```bash
## runSwarm.sh

# Build and tag the consumer image
docker build -t kafka-consumer ./kafka_consumer

# Create the Swarm
docker swarm init
docker network create --attachable -d overlay swarm

# Instantiate graphite + grafana and bind ports
docker service create --name graphite --publish 8080:80 --network swarm graphiteapp/graphite-statsd
docker service create --name grafana --publish 80:3000 grafana/grafana

# Run the Kafka consumer in the same network
docker run -d --name consumer --hostname kafka --network swarm kafka-consumer
```
After executing this, the Graphite instance would be consuming from the specified Kafka server + topic and the UI would be available at port 8080. Grafana UI should be accesible at port 80 and it should allow a connection with Graphite with the hostname 'graphite'.