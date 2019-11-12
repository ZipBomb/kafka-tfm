package com.tfm.ksqlmlflowtest;

import java.text.MessageFormat;

import io.confluent.ksql.function.udf.Udf;
import io.confluent.ksql.function.udf.UdfDescription;

import kong.unirest.Unirest;


@UdfDescription(name = "predict", description = "call mlflow endpoint to get a prediction based on the current " +
    "flotation plant state")
public class MiningPredictionUDF {

    @Udf(description = "get prediction given current flotation plant state")
    double predict(double dayOfMonth, double dayOfWeek, double hour, double ironFeed, double starchFlow, double aminaFlow,
                   double orePulpFlow, double orePulpPh, double orePulpDensity,
                   double fc1af, double fc2af, double fc3af, double fc4af, double fc5af, double fc6af, double fc7af,
                   double fc1l, double fc2l, double fc3l, double fc4l, double fc5l, double fc6l, double fc7l, double scLag2) {

        String toFormat = "'{'\"columns\":[\"day_of_month\",\"day_of_week\",\"hour\",\"% Iron Feed\",\"Starch Flow\",\"Amina Flow\"," +
            "\"Ore Pulp Flow\",\"Ore Pulp pH\",\"Ore Pulp Density\",\"Flotation Column 01 Air Flow\",\"Flotation Column 02 Air Flow\"," +
            "\"Flotation Column 03 Air Flow\",\"Flotation Column 04 Air Flow\",\"Flotation Column 05 Air Flow\"," +
            "\"Flotation Column 06 Air Flow\",\"Flotation Column 07 Air Flow\",\"Flotation Column 01 Level\"," +
            "\"Flotation Column 02 Level\",\"Flotation Column 03 Level\",\"Flotation Column 04 Level\"," + 
            "\"Flotation Column 05 Level\",\"Flotation Column 06 Level\",\"Flotation Column 07 Level\",\"sc_lag2\"]," +
            "\"data\":[[{0,number,#0.0},{1,number,#0.0},{2,number,#0.0},{3},{4,number,#0.000},{5},{6},{7},{8},{9},{10},{11},{12},{13}," +
            "{14},{15},{16},{17},{18},{19},{20},{21},{22},{23}]]'}'";

        String formatted = MessageFormat.format(toFormat, dayOfMonth, dayOfWeek, hour, ironFeed, starchFlow, aminaFlow, orePulpFlow, orePulpPh, orePulpDensity,
            fc1af, fc2af, fc3af, fc4af, fc5af, fc6af, fc7af, fc1l, fc2l
                , fc3l, fc4l, fc5l, fc6l, fc7l, scLag2);

        String ENDPOINT = "http://<MLFLOW_SERVER>:5000/invocations";
        String response = Unirest.post(ENDPOINT)
            .header("Content-Type", "application/json")
            .header("format", "pandas-split")
            .body(formatted)
            .asString()
            .getBody();

        return new Double(response.substring(response.indexOf("[") + 1, response.indexOf("]")));
    }

}