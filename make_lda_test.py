# coding: utf-8

from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, Tokenizer

messages = sc.textFile("/Users/faisal/Projects/eaf_v2/eaf_spark/data/examples.txt").flatMap(lambda x: x.split('\n'))
messages.collect()

from pyspark.sql import Row

messages.zipWithIndex().map(lambda (body,docID): Row(docID=docID, body=body)).toDF()
msg_df = messages.zipWithIndex().map(lambda (body,docID): Row(docID=docID, body=body)).toDF()
msg_df.show()
toker = Tokenizer(inputCol='body', outputCol='tokens')
cver = CountVectorizer(inputCol='tokens', outputCol='vec')
msg_tok_df = toker.transform(msg_df)
cvmodel = cver.fit(msg_tok_df)
msg_vec = cvmodel.transform(msg_tok_df).select('docID', 'body', 'vec')
msg_vec.show()

from pyspark.mllib.linalg import Vector, SparseVector, DenseVector

corpus = msg_vec.rdd.map(lambda row: [int(row['docID']), DenseVector(row.asDict()['vec'].toArray())])
corpus.collect()
ldaModel = LDA.train(corpus, k=5)
p = ldaModel.topicDistributions().collect()

print p
