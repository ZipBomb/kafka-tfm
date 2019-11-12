package com.tfm.ksqlmlflowtest;

import org.junit.Test;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertEquals;

import kong.unirest.Unirest;
import kong.unirest.JsonNode;
import kong.unirest.HttpResponse;
import kong.unirest.UnirestException;

/**
 * Unit test for simple App.
 */
public class AppTest {

    /**
     * Try the Unirest HTTP client
     */
    @Test
    public void shouldReturnStatusOkay() throws UnirestException {
        HttpResponse <JsonNode> jsonResponse = Unirest.get("http://www.mocky.io/v2/5a9ce37b3100004f00ab5154")
            .header("accept", "application/json")
            .asJson();

        assertNotNull(jsonResponse.getBody());
        assertEquals(200, jsonResponse.getStatus());
    }
    /**
     * Check if mlflow remote server provides the correct answer
     */
    @Test
    public void shouldAnswerWithTrue() throws UnirestException {
        String kafkaServer = "<KAFKA_SERVER_URI>";
        String response = Unirest.post(kafkaServer)
            .header("Content-Type", "application/json")
            .header("format", "pandas-split")
            .body("{\"columns\":[\"day_of_month\",\"day_of_week\",\"hour\",\"% Iron Feed\",\"Starch Flow\",\"Amina Flow\",\"Ore Pulp Flow\"," +
                    "\"Ore Pulp pH\",\"Ore Pulp Density\",\"Flotation Column 01 Air Flow\",\"Flotation Column 02 Air Flow\",\"Flotation Column 03 Air Flow\"," +
                    "\"Flotation Column 04 Air Flow\",\"Flotation Column 05 Air Flow\",\"Flotation Column 06 Air Flow\",\"Flotation Column 07 Air Flow\"," +
                    "\"Flotation Column 01 Level\",\"Flotation Column 02 Level\",\"Flotation Column 03 Level\",\"Flotation Column 04 Level\"," +
                    "\"Flotation Column 05 Level\",\"Flotation Column 06 Level\",\"Flotation Column 07 Level\",\"sc_lag2\"]," +
                    "\"data\":[[10.0,4.0,1.0,55.2,3162.6250258621,578.7866781609,398.7533678161,10.1134873563,1.7295581609," +
                    "251.1666724138,250.2260862069,250.1782873563,295.096,306.4,251.2325287356,250.208183908,450.3837758621," +
                    "446.8918448276,450.4745229885,449.9122586207,455.7921609195,464.3833103448,450.5327471264,2.326993895]]}")
            .asString()
            .getBody();

        assertEquals("[1.5824519230769243]", response);
    }
    /**
     * Check if mlflow remote server provides the correct answer
     */
    @Test
    public void shouldAnswerWithTrueToo() throws UnirestException {
        double response = new MiningPredictionUDF().predict(10.0, 4.0, 1.0, 55.2, 3162.6250258621,
                578.7866781609, 398.7533678161, 10.1134873563, 1.7295581609, 251.1666724138,
                250.2260862069, 250.1782873563, 295.096, 306.4, 251.2325287356, 250.208183908, 450.3837758621,
                446.8918448276, 450.4745229885, 449.9122586207, 455.7921609195,
                464.3833103448, 450.5327471264, 2.326993895);

        assertEquals(1.5824519230769243, response, 0.0);
    }    
    /**
     * Check if mlflow remote server provides the correct answer
     */
    @Test
    public void shouldAnswerWithTrueTooToo() throws UnirestException {
        double response = new MiningPredictionUDF().predict(12.0, 3.0, 4.0, 55.6, 1564.6250258621,
                236.7866781609, 485.7533678161, 11.1134873563, 1.7295581609, 400.1666724138,
                121.2260862069, 365.1782873563, 26.096, 789.4, 251.2325287356, 250.208183908, 450.3837758621,
            665.8918448276, 789.4745229885, 132.9122586207, 77.7921609195, 444.3833103448, 450.5327471264, 1.17);

        assertTrue(response != 1.5824519230769243);
    }
}