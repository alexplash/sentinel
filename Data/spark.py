from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName('sentinel') \
    .getOrCreate()

df1 = spark.read.option('header', 'true').csv('/Users/alexplash/sentinel/Data/Youtube/Youtube01-Psy.csv')
df2 = spark.read.option('header', 'true').csv('/Users/alexplash/sentinel/Data/Youtube/Youtube02-KatyPerry.csv')
df3 = spark.read.option('header', 'true').csv('/Users/alexplash/sentinel/Data/Youtube/Youtube03-LMFAO.csv')
df4 = spark.read.option('header', 'true').csv('/Users/alexplash/sentinel/Data/Youtube/Youtube04-Eminem.csv')
df5 = spark.read.option('header', 'true').csv('/Users/alexplash/sentinel/Data/Youtube/Youtube05-Shakira.csv')
youtube_df = df1.union(df2).union(df3).union(df4).union(df5)
youtube_df = youtube_df.filter(youtube_df['CLASS'].isin(['1', '0']))

df6 = spark.read.option('header', 'true').csv('/Users/alexplash/sentinel/Data/twitter spam data.csv')
twitter_df = df6.withColumnRenamed('class', 'CLASS') \
                  .withColumnRenamed('tweets', 'CONTENT') \
                  .withColumn('COMMENT_ID', F.lit(None)) \
                  .withColumn('AUTHOR', F.lit(None)) \
                  .withColumn('DATE', F.lit(None)) \
                  .select('COMMENT_ID', 'AUTHOR', 'DATE', 'CONTENT', 'CLASS')
twitter_df = twitter_df.filter(twitter_df['CLASS'].isin(['1', '0']))

df7 = spark.read.option('header', 'true').csv('/Users/alexplash/sentinel/Data/SMS_spam.csv')
sms_df = df7.withColumn('v1', F.when(F.col('v1') == 'spam', '1').otherwise('0')) \
         .withColumnRenamed('v1', 'CLASS') \
         .withColumnRenamed('v2', 'CONTENT') \
         .withColumn('COMMENT_ID', F.lit(None)) \
         .withColumn('AUTHOR', F.lit(None)) \
         .withColumn('DATE', F.lit(None)) \
         .select('COMMENT_ID', 'AUTHOR', 'DATE', 'CONTENT', 'CLASS')

df8 = spark.read.option('header', 'true').csv('/Users/alexplash/sentinel/Data/other_spam.csv')
other_df = df8.withColumnRenamed('label', 'CLASS') \
              .withColumnRenamed('text', 'CONTENT') \
              .withColumn('COMMENT_ID', F.lit(None)) \
              .withColumn('AUTHOR', F.lit(None)) \
              .withColumn('DATE', F.lit(None)) \
              .select('COMMENT_ID', 'AUTHOR', 'DATE', 'CONTENT', 'CLASS')
other_df = other_df.filter(other_df['CLASS'].isin(['1', '0']))

df = youtube_df.union(twitter_df).union(sms_df).union(other_df)

df = df.drop('COMMENT_ID', 'AUTHOR', 'DATE')
df.na.drop(subset = ['CONTENT', 'CLASS'])
df.dropDuplicates()

df.coalesce(1).write.json(path = '/Users/alexplash/sentinel/Data/processed_data.json', mode = 'overwrite')