'''

https://spark.apache.org/docs/latest/ml-classification-regression.html

'''

import pyspark
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

sc = SparkContext("local")
spark = SparkSession.builder.getOrCreate()


##########

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

data = [
(Vectors.dense([4.0, 1.1]),),
(Vectors.dense([6.0, 1.1]),),
(Vectors.dense([7.1, 1.2]),),
(Vectors.dense([8.1, 1.3]),),
(Vectors.dense([9.1, 1.1]),),
]
df = spark.createDataFrame(data, ["features"])

r1 = Correlation.corr(df, "features").head()

print("Pearson correlation matrix:\n" + str(r1[0]))

##########

'''

wget https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_libsvm_data.txt

'''

from pyspark.ml.classification import LogisticRegression

training = spark.read.format("libsvm").load("sample_libsvm_data.txt")

lr = LogisticRegression(
	maxIter=10, 
	regParam=0.3,
	elasticNetParam=0.8)

lrModel = lr.fit(training)

prediction = lrModel.transform(training)

prediction.registerTempTable("prediction")

spark.sql(u"""
	SELECT label, prediction, COUNT(*)
	FROM prediction
	GROUP BY label, prediction
	""").show()


############

'''

wget https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_linear_regression_data.txt

'''

from pyspark.ml.regression import GeneralizedLinearRegression

training = spark.read.format("libsvm").load("sample_linear_regression_data.txt")

glr = GeneralizedLinearRegression(
	family="gaussian", link="identity", maxIter=2000, regParam=0.3)

model = glr.fit(training)

prediction = model.transform(training)


############


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

'''

wget https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_kmeans_data.txt

'''

dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")

kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

predictions = model.transform(dataset)

############