-- RAW_DATA STREAM CREATION
CREATE STREAM raw_data (
    date STRING, 
    day_of_month DOUBLE,
    day_of_week DOUBLE,
    hour DOUBLE,
    iron_feed DOUBLE,
    starch_flow DOUBLE,
    amina_flow DOUBLE,
    ore_pulp_flow DOUBLE,
    ore_pulp_ph DOUBLE,
    ore_pulp_density DOUBLE,
    fc1af DOUBLE,
    fc2af DOUBLE,
    fc3af DOUBLE,
    fc4af DOUBLE,
    fc5af DOUBLE,
    fc6af DOUBLE,
    fc7af DOUBLE,
    fc1l DOUBLE,
    fc2l DOUBLE,
    fc3l DOUBLE,
    fc4l DOUBLE,
    fc5l DOUBLE,
    fc6l DOUBLE,
    fc7l DOUBLE
) WITH (kafka_topic='raw_data', value_format='json', key='date');

-- LAB_DATA TABLE CREATION
CREATE TABLE lab_data (
    date STRING, 
    silica_concentrate DOUBLE
) WITH (kafka_topic='lab_data', value_format='json', key='date');

-- CREATE THE RAW + LAB DATA STREAM
CREATE STREAM RAW_WITH_LAB_DATA AS
SELECT 
    r.day_of_month,
    r.day_of_week,
    r.hour,
    r.iron_feed,
    r.starch_flow,
    r.amina_flow,
    r.ore_pulp_flow,
    r.ore_pulp_ph,
    r.ore_pulp_density,
    r.fc1af,
    r.fc2af,
    r.fc3af,
    r.fc4af,
    r.fc5af,
    r.fc6af,
    r.fc7af,
    r.fc1l,
    r.fc2l,
    r.fc3l,
    r.fc4l,
    r.fc5l,
    r.fc6l,
    r.fc7l,   
    l.silica_concentrate
FROM raw_data r LEFT JOIN lab_data l ON r.date = l.date;

CREATE STREAM RAW_AND_PREDICTED AS 
SELECT 
    day_of_month,
    day_of_week,
    hour,
    iron_feed,
    starch_flow,
    amina_flow,
    ore_pulp_flow,
    ore_pulp_ph,
    ore_pulp_density,
    fc1af,
    fc2af,
    fc3af,
    fc4af,
    fc5af,
    fc6af,
    fc7af,
    fc1l,
    fc2l,
    fc3l,
    fc4l,
    fc5l,
    fc6l,
    fc7l,   
    silica_concentrate, 
    PREDICT(day_of_month, day_of_week, hour, 
        iron_feed, starch_flow, amina_flow, 
        ore_pulp_flow, ore_pulp_ph, ore_pulp_density, 
        fc1af, fc2af, fc3af, fc4af, fc5af, fc6af, fc7af, 
        fc1l, fc2l, fc3l, fc4l, fc5l, fc6l, fc7l, silica_concentrate) AS predicted 
FROM RAW_WITH_LAB_DATA;

-- TERMINATE CSAS_RAW_AND_PREDICTED_1;
-- DROP STREAM RAW_AND_PREDICTED;