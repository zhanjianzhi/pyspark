#This Python file uses the following encoding: utf-8

from __future__ import print_function
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import HiveContext
import re
from functools import reduce
import datetime
from pyspark import SparkContext,SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import udf, col

#常规配置
conf = SparkConf().setAppName("05-test")
sc = SparkContext(conf=conf)
sqlContext = HiveContext(sc)
spark = SparkSession(sc)

#引入时间（前一天）
now_time = datetime.datetime.now()
yes_time = now_time + datetime.timedelta(days=-1)
date = yes_time.strftime('%Y-%m-%d')
#date = '2018-07-07'
#处理规则
chineseRe = re.compile(u'[\u4e00-\u9fa5]')
numRe = re.compile(r'\d+|\%|\\.|~|\+')
titleLen = 6
contentLen=120
wordNum=82
words = u"I'm QQ - 每一天，乐在沟通"
#每日用户爬取数据
referPath = 'hdfs://nameservice1/user/hive/warehouse/xxzx.db/url_result/refer_'+date + '*'
#直接读取与分步读取
try:
    referDataDF = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(referPath)
except:
    referData = sc.textFile(referPath)
    headData = referData.first()
    referData01 = referData.filter(lambda x:x != headData).map(lambda x:x.split(',')).filter(lambda x:len(x) == 6)
    referDataDF = spark.createDataFrame(referData01,headData.split(','))

#定义字段名
colName01 = referDataDF.columns + ['contentTemp']
#题出标题和内容有一个为空的行
referDataDFAllTemp01 = referDataDF.rdd.map(list).filter(lambda x:(x[4] != None) and (x[5] != None)).map(lambda x:[x[0],x[1],x[2],x[3],x[4],x[5],x[5][:wordNum]])
#过滤无效行
referDataDFAllTemp = referDataDFAllTemp01.filter(lambda x:(x[1] != None) & (len(x[4]) > titleLen) & (len(x[5]) > contentLen) & (x[4] != words))
#构造数据框
referDataDFAllTemp = spark.createDataFrame(referDataDFAllTemp,colName01)
#定义选取列
selectCol = ['instId', 'title', 'content', 'host','contentTemp']
#选取指定列
referData01 = referDataDFAllTemp.select(selectCol)
#以用户和文章前82个字符去重
referData02 = referData01.dropDuplicates(subset=['instId','contentTemp'])
#drop零时列
referData02 = referData02.drop('contentTemp')
#正则化数字与特殊符号
referData03 = referData02.rdd.map(list).map(lambda x:[x[0],re.sub(numRe, '', x[1]),re.sub(numRe, '', x[2]),x[3]])
#定义字段名
colNew = list(referData02.columns)
#构造数据框
referDataToDF = spark.createDataFrame(referData03,colNew)
#存储文件
referDataToDF.write.csv('/user/xxzx/userdpi/output/output_%s/05-referDataPro'%date,header = True)
