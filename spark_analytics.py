from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, min, max, count, col, when
from pyspark.sql.functions import round as spark_round
import time

spark = SparkSession.builder \
    .appName("HousingSparkAnalytics") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

t_start = time.time()
df = spark.read.csv("/opt/data/housing_2.csv", header=True, inferSchema=True)
t_load = time.time() - t_start

print("rows:", df.count())
print("cols:", df.columns)
print()

# avg house value by proximity
t0 = time.time()
q1 = df.groupBy("ocean_proximity") \
    .agg(
        count("*").alias("record_count"),
        spark_round(avg("median_house_value"), 2).alias("avg_house_value"),
        spark_round(min("median_house_value"), 2).alias("min_house_value"),
        spark_round(max("median_house_value"), 2).alias("max_house_value")
    ) \
    .orderBy(col("avg_house_value").desc())
q1.show()
t1 = time.time() - t0

# income brackets
t0 = time.time()
q2 = df.withColumn(
    "income_bracket",
    when(col("median_income") < 2.0, "Low (under 2)")
    .when(col("median_income") < 4.0, "Medium (2 to 4)")
    .when(col("median_income") < 6.0, "High (4 to 6)")
    .otherwise("Very High (over 6)")
).groupBy("income_bracket") \
 .agg(
    count("*").alias("count"),
    spark_round(avg("median_house_value"), 2).alias("avg_house_value"),
    spark_round(avg("median_income"), 4).alias("avg_income")
 ) \
 .orderBy("avg_income")
q2.show()
t2 = time.time() - t0

# housing age -- older stock near bay
t0 = time.time()
q3 = df.groupBy("ocean_proximity") \
    .agg(
        spark_round(avg("housing_median_age"), 2).alias("avg_age"),
        spark_round(min("housing_median_age"), 2).alias("min_age"),
        spark_round(max("housing_median_age"), 2).alias("max_age"),
        count("*").alias("count")
    ) \
    .orderBy(col("avg_age").desc())
q3.show()
t3 = time.time() - t0

# inland only
t0 = time.time()
q4 = df.filter(col("ocean_proximity") == "INLAND") \
    .agg(
        spark_round(avg("median_house_value"), 2).alias("avg_inland_value"),
        spark_round(avg("median_income"), 4).alias("avg_inland_income"),
        count("*").alias("count")
    )
q4.show()
t4 = time.time() - t0

# top 10 priciest
t0 = time.time()
q5 = df.select(
    "longitude", "latitude", "ocean_proximity",
    "housing_median_age", "median_income", "median_house_value"
).orderBy(col("median_house_value").desc()).limit(10)
q5.show()
t5 = time.time() - t0

print("load:", t_load, "q1:", t1, "q2:", t2, "q3:", t3, "q4:", t4, "q5:", t5,
      "total:", t_load + t1 + t2 + t3 + t4 + t5)

spark.stop()