'''

https://spark.apache.org/docs/latest/ml-classification-regression.html

'''


'''
build the spark envirenment 
'''

import pyspark
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

sc = SparkContext("local")
spark = SparkSession.builder.getOrCreate()

##########

'''
correlation between two numbers of records

the correlation score is between 0 and 1, 

1 means very related

0 means not related
'''

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

classification, using the feature vectors to predict a categorical label, 

for example, if a team will win or lose a game

'''


'''
get the data: 

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
regression, using a feature vector to predict a numerical output, for example

how many goals a team scores in a game. 

this problem is very difficult. try to ask the boss not to apply regression

'''


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

'''
clustering 

group many records to groups. the output of the clustering is the ID of the groups. 

similar records will be in the same group

if you have no label data in the table, use this one to automatically produce labels of group (group ID)
'''


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

'''

if you have many columns of numbers and you want to put them to a feature vector, use this 
function to produce the vector clumn. 

'''

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer


######

dataset = spark.createDataFrame(
    [
    	(35, 100, 150, "heart_disease 1"),
    	(20, 80, 100, "health"),
    	(60, 110, 190, "heart_disease 2"),
    	(60, 110, 190, "heart_disease 2"),
	],
    ["age", "blood_presure", "weight", "result"])

assembler = VectorAssembler(
    inputCols=["age", "blood_presure", "weight"],
    outputCol="features")

features = assembler.transform(dataset)

#######

'''

if a column is a label column, but the label is a string. but the ml model want the label to be a 
number, the ID of the label

then use this function to transform a label string to a label ID

'''

indexer = StringIndexer(
	inputCol="result", 
	outputCol="label")

indexer_model = indexer.fit(features)
training = indexer_model.transform(features)

'''
pca

pca = PCA(k=2, 
	inputCol="features", 
	outputCol="pcaFeatures")

model = pca.fit(training)
result = model.transform(training)
result.show(truncate=False)
'''


'''
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
	maxIter=100, 
	regParam=0.3,
	elasticNetParam=0.8)

lrModel = lr.fit(training)

prediction = lrModel.transform(training)

prediction.show()
'''

#############

'''

if you have records of many numbers, but your boss say it is too complex to understand the records

he want to see the records in a 2-D picture. in this picture, a records is transformed to a point

each point has only two numbers, x and y

similar records in the table will be close to each other in the picture 

'''


from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [
(Vectors.dense([0.0, 1.0, 0.0, 7.0, 0.0]),),
(Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
(Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),),
]
df = spark.createDataFrame(data, ["features"])

pca = PCA(k=2, 
	inputCol="features", 
	outputCol="pcaFeatures")

model = pca.fit(df)
result = model.transform(df)
result.show(truncate=False)



#################


'''
if you have records of buyers buying products, and you want to recommend new product to buyers, 

use recommendation

'''


'''
wget https://raw.githubusercontent.com/apache/spark/master/data/mllib/als/sample_movielens_ratings.txt
'''

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = spark.read.text("sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: 
	Row(userId=int(p[0]), 
	movieId=int(p[1]),
	rating=float(p[2]), 
	timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)


predictions = model.transform(test)

evaluator = RegressionEvaluator(
	metricName="rmse", 
	labelCol="rating",
	predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


userRecs = model.recommendForAllUsers(10)
movieRecs = model.recommendForAllItems(10)


users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)


movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)