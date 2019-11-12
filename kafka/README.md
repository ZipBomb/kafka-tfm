## IMPORTANT: After Kafka cluster restart set again 'advertised.listeners' property on '~/confluent-5.2.2/etc/kafka/server.properties' and execute 'confluent start'. Confluent Center port is 9021.

## Prediction test

### Server
```bash
mlflow pyfunc serve -r 3981ed6d225f40b9b101103216d00f7e -m model -h 0.0.0.0
```

### Client
```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["day_of_month","day_of_week","hour","% Iron Feed","Starch Flow","Amina Flow","Ore Pulp Flow","Ore Pulp pH","Ore Pulp Density","Flotation Column 01 Air Flow","Flotation Column 02 Air Flow","Flotation Column 03 Air Flow","Flotation Column 04 Air Flow","Flotation Column 05 Air Flow","Flotation Column 06 Air Flow","Flotation Column 07 Air Flow","Flotation Column 01 Level","Flotation Column 02 Level","Flotation Column 03 Level","Flotation Column 04 Level","Flotation Column 05 Level","Flotation Column 06 Level","Flotation Column 07 Level","sc_lag2"],"data":[[10.0,4.0,1.0,55.2,3162.6250258621,578.7866781609,398.7533678161,10.1134873563,1.7295581609,251.1666724138,250.2260862069,250.1782873563,295.096,306.4,251.2325287356,250.208183908,450.3837758621,446.8918448276,450.4745229885,449.9122586207,455.7921609195,464.3833103448,450.5327471264,2.326993895]]}' http://<MLFLOW_TRACKING_URI>:5000/invocations
```
