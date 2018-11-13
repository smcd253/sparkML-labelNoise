from pyspark.sql import SparkSession

from pyspark.ml.clustering import KMeans

train_pro = spark.read.csv("train_pro.tsv", header = True, inferSchema = True)

train_pro.describe().show()
