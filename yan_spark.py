import pyspark
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

sc = SparkContext("local")
sqlContext = SparkSession.builder.getOrCreate()

'''
sqlContext = SparkSession.builder\
	.master('local[*]') \
	.config("spark.driver.memory", "30g")\
	.appName('yan') \
	.getOrCreate()
'''
