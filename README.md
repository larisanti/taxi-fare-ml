# Taxi Fare Forecasting using BigQuery ML

This project is a lab exercise completed during the Machine Learning Engineer Learning Path course. It demonstrates a sequence of BigQuery ML queries to build and evaluate a linear regression model for predicting New York City taxi fares based on trip data.

The project utilizes the `bigquery-public-data.new_york.tlc_yellow_trips_2015` dataset, focusing on features like pickup and dropoff locations, time of day, and passenger count to predict the total fare amount (including tolls). Through model training, evaluation, and refinement techniques such as feature engineering, the project showcases the capabilities of BigQuery ML for real-world applications like fare estimation.

## Workflow

### 1.  **Explore the dataset: Calculate the number of trips**
Run the query [`trips_per_month.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/trips_per_month.sql) to calculate the number of trips Yellow taxis took each month in 2015 from the NYC Yellow taxi trip dataset. The query groups the data by month and presents the results in chronological order.

### 2.  **Explore the dataset: Calculate the average speed**
Run the query [`speed_per_hour.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/speed_per_hour.sql) to calculate the average speed of Yellow taxi trips for each hour of the day in 2015, considering only trips with valid distance, fare-to-distance ratios, and pickup/dropoff times.

### 3.  **Prepare the training data**
This step (query: [`training_data.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/training_data.sql)) prepares the training data for a taxi fare prediction model. It selects relevant features, performs feature engineering (calculating `total_fare`, extracting day of the week and hour of the day), filters out invalid data, and splits the data into training and evaluation sets using hashing.


### 4.  **Create and train the taxifare_model**
   
```sql
CREATE or REPLACE MODEL taxi.taxifare_model
OPTIONS
  (model_type='linear_reg', labels=['total_fare']) AS -- specify the model type: linear regression

WITH params AS (
    SELECT
    1 AS TRAIN,
    2 AS EVAL
    ),

  daynames AS
    (SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek),

  taxitrips AS (
  SELECT
    (tolls_amount + fare_amount) AS total_fare,
    daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
    EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
    pickup_longitude AS pickuplon,
    pickup_latitude AS pickuplat,
    dropoff_longitude AS dropofflon,
    dropoff_latitude AS dropofflat,
    passenger_count AS passengers
  FROM
    `nyc-tlc.yellow.trips`, daynames, params
  WHERE
    trip_distance > 0 AND fare_amount > 0
    AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.TRAIN
  )

  SELECT *
  FROM taxitrips
```

![ss 3]()

*(query in: [`taxifare_model.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/taxifare_model.sql))*

### 5.  **Evaluate the model using ML.EVALUATE**
   
```sql
SELECT
  SQRT(mean_squared_error) AS rmse
FROM
  ML.EVALUATE(MODEL taxi.taxifare_model,
  (

  WITH params AS (
    SELECT
    1 AS TRAIN,
    2 AS EVAL
    ),

  daynames AS
    (SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek),

  taxitrips AS (
  SELECT
    (tolls_amount + fare_amount) AS total_fare,
    daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
    EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
    pickup_longitude AS pickuplon,
    pickup_latitude AS pickuplat,
    dropoff_longitude AS dropofflon,
    dropoff_latitude AS dropofflat,
    passenger_count AS passengers
  FROM
    `nyc-tlc.yellow.trips`, daynames, params
  WHERE
    trip_distance > 0 AND fare_amount > 0
    AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.EVAL
  )

  SELECT *
  FROM taxitrips

  ))
```

![ss 4]()

*(query in: [`evaluate_taxifare_model.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/evaluate_taxifare_model.sql))*

### 6.  **Predict taxi fares**
In this step (query [`predict_taxi_fare.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/predict_taxi_fare.sql)), the taxi.taxifare_model is used to predict taxi fare amounts. The query applies the model to a subset of the `nyc-tlc.yellow.trips` dataset, generating predictions based on trip features such as pickup/dropoff locations, time of day, and passenger count.


### 7.  **Improve the model with Feature Engineering**
This step focuses on improving the taxi fare prediction model by using feature engineering and data filtering techniques. The goal is to identify and select the most relevant features and data points for training a more accurate model. This is achieved by running a sequence of three queries:
* 7.1. Explore initial fare statistics to understand the overall fare distribution
* 7.2. Filter out-of-range fares to avoid learning on outliers
* 7.3. Limit to NYC geographic coordinates to ensure that the model is trained on trips that are relevant to the target area
 *(complete sequency of queries here: [`feature_engineering.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/feature_engineering.sql))*


### 8.  **Retrain the model / create taxi.taxifare_model_2**
Next, this query [`taxifare_model_2.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/taxifare_model_2.sql) helps to improve the accuracy of the taxi fare prediction model by incorporating new features, filtering out irrelevant data, and retraining the model using a refined dataset.


### 9.  **Evaluate taxifare_model_2**
   
```sql
SELECT
  SQRT(mean_squared_error) AS rmse
FROM
  ML.EVALUATE(MODEL taxi.taxifare_model_2,
  (

  WITH params AS (
    SELECT
    1 AS TRAIN,
    2 AS EVAL
    ),

  daynames AS
    (SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek),

  taxitrips AS (
  SELECT
    (tolls_amount + fare_amount) AS total_fare,
    daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
    EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
    SQRT(POW((pickup_longitude - dropoff_longitude),2) + POW(( pickup_latitude - dropoff_latitude), 2)) as dist, #Euclidean distance between pickup and drop off
    SQRT(POW((pickup_longitude - dropoff_longitude),2)) as longitude, #Euclidean distance between pickup and drop off in longitude
    SQRT(POW((pickup_latitude - dropoff_latitude), 2)) as latitude, #Euclidean distance between pickup and drop off in latitude
    passenger_count AS passengers
  FROM
    `nyc-tlc.yellow.trips`, daynames, params
WHERE trip_distance > 0 AND fare_amount BETWEEN 6 and 200
    AND pickup_longitude > -75 #limiting of the distance the taxis travel out
    AND pickup_longitude < -73
    AND dropoff_longitude > -75
    AND dropoff_longitude < -73
    AND pickup_latitude > 40
    AND pickup_latitude < 42
    AND dropoff_latitude > 40
    AND dropoff_latitude < 42
    AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),1000) = params.EVAL
  )

  SELECT *
  FROM taxitrips

  ))
```

![ss 7]()

*(query in: [`evaluate_taxifare_model_2.sql`](https://github.com/larisanti/taxi-forecasting-ml/blob/main/evaluate_taxifare_model_2.sql))*
