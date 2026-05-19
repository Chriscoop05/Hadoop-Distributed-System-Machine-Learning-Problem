from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import time #time is Pythons timer to measure how long each model takes to train

#Starts the spark session 
spark = SparkSession.builder \
    .appName("HousingMLPipeline") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

#Reads the data from the HDFS into a spark dataframe 
df = spark.read.csv(
    "hdfs://namenode:9000/user/housing/raw/housing_2.csv",
    header=True,
    inferSchema=True    #spark figures out the data types automatically instead of treating everything as strings
)

print("rows:", df.count(), "cols:", len(df.columns))
df.show(5)

# impute missing bedrooms values with the median values of the non-null elements 
imputer = Imputer(
    inputCols=["total_bedrooms"],
    outputCols=["total_bedrooms_imputed"],
    strategy="median"
)
df_imputed = imputer.fit(df).transform(df)

# encode ocean_proximity from string to factor values 
string_indexer = StringIndexer(inputCol="ocean_proximity", outputCol="ocean_index", handleInvalid="keep") #converts the ocean_prox values to dummy vals...handleinvalid makes sure to label any unknown category as "unknown" instead of crashing 
encoder = OneHotEncoder(inputCols=["ocean_index"], outputCols=["ocean_vec"], dropLast=True) #One-hot encodes the new dummy vals and drops last column to avid multi-collinearity 

feature_cols = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms_imputed", "population",
    "households", "median_income", "ocean_vec"
]

#Assemble all features into one vector so that pyspark ml's models can read the input feature values .... handleInvalid = false gets rid of any rows that are still NULL after imputation 
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
#Scales all features on a scale of mean 0 with STD of 1
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True) 



#Splits the data into training/testing data...80% = training while 20% is used for testing
train_df, test_df = df_imputed.randomSplit([0.8, 0.2], seed=123)
print("train:", train_df.count(), "test:", test_df.count())

# normal equation without any parameterization (regParam = 0 and elsasticNetParam = 0)
lr_normal = LinearRegression(featuresCol="features", labelCol="median_house_value",
    solver="normal", regParam=0.0, elasticNetParam=0.0)
#Pipelines all preprocessing steps together with the training/fitting of the model in one swoop.
pipeline_normal = Pipeline(stages=[string_indexer, encoder, assembler, scaler, lr_normal])
#tracks how long the time takes to run each model to get the computation performance 
t0 = time.time()
#fits the model 
model_normal = pipeline_normal.fit(train_df)
t_normal = time.time() - t0



# gradient descent: 100 iterations instead of one-shot...reduces the loss function over iterations on a gradient in order to find the most optimal coefficients 
lr_gd = LinearRegression(featuresCol="features", labelCol="median_house_value",
    solver="l-bfgs", maxIter=100, regParam=0.0, elasticNetParam=0.0)
pipeline_gd = Pipeline(stages=[string_indexer, encoder, assembler, scaler, lr_gd])
t0 = time.time()
model_gd = pipeline_gd.fit(train_df)
t_gd = time.time() - t0

#ridge regression - good at handling multicollinearity...squeezes coefficients down so no feature can dominate 
#Use ridge regression when you are dealing with multicollinearity and want to prevent overfitting 
lr_ridge = LinearRegression(featuresCol="features", labelCol="median_house_value",
    solver="l-bfgs", maxIter=100, regParam=0.1, elasticNetParam=0.0) #reg param = 0.1 applies slight nudge to keep coefficients small
pipeline_ridge = Pipeline(stages=[string_indexer, encoder, assembler, scaler, lr_ridge])
t0 = time.time()
model_ridge = pipeline_ridge.fit(train_df)
t_ridge = time.time() - t0

#lasso regression - gets rid of useless features and reduces them to 0
#typically use lasso when high dimensional data and you want to simplify the inputs and seek simplification
lr_lasso = LinearRegression(featuresCol="features", labelCol="median_house_value",
    solver="l-bfgs", maxIter=100, regParam=0.1, elasticNetParam=1.0) #while reg param is the strengh of penalty, elsasticNet is whether it's Lasso/Ridge
pipeline_lasso = Pipeline(stages=[string_indexer, encoder, assembler, scaler, lr_lasso])
t0 = time.time()
model_lasso = pipeline_lasso.fit(train_df)
t_lasso = time.time() - t0


#Calculate the RMSE and R^2 to get the error and explained variance from the model
rmse_eval = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="rmse")
r2_eval = RegressionEvaluator(labelCol="median_house_value", predictionCol="prediction", metricName="r2")

#Prep our models into a dict together so that we can loop over each of the four model evals 
models = {
    "normal": model_normal,
    "gd": model_gd,
    "ridge": model_ridge,
    "lasso": model_lasso
}
times = {"normal": t_normal, "gd": t_gd, "ridge": t_ridge, "lasso": t_lasso}

results = {}
for name, model in models.items():
    preds = model.transform(test_df)
    results[name] = {"rmse": rmse_eval.evaluate(preds), "r2": r2_eval.evaluate(preds)}

# sample preds vs their real values 
model_normal.transform(test_df).select(
    "ocean_proximity", "median_income", "housing_median_age",
    "median_house_value", "prediction"
).show(10)

#Gets all the errors and eval metrics for all four models to compare 
for name, metrics in results.items():
    print(name, "rmse:", round(metrics["rmse"], 2), "r2:", round(metrics["r2"], 4), "time:", round(times[name], 2))

spark.stop()