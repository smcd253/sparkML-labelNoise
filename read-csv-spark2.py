from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.appName('genderClassifier').getOrCreate()

train_pro = spark.read.csv("spencer/sum_tab_1.csv", header = True, inferSchema = True)

train_pro.head()
