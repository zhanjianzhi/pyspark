# -*- coding: utf-8 -*-
import datetime
import random
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.ml import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.regression import *
from pyspark.ml.clustering import *
from pyspark.ml.recommendation import *
from pyspark.ml.evaluation import *

SPLIT_FLAG = "|"
VECTOR_FEATURE = "vector_features"

def unique():
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    randomNum = random.randint(0, 100)
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)
    return str(nowTime) + str(randomNum)

def mlp_dbHive(**params):
    spark = params.get("spark")
    dbName = params.get("dbName")
    tableName = params.get("tableName")
    columnName = params.get("columnName")
    partitionFlag = params.get("partitionFlag")
    partitionCondition = params.get("partitionCondition")
    sql_list = ["select"]
    sql_list.append(columnName.replace(SPLIT_FLAG,","))
    sql_list.append("from")
    sql_list.append(dbName+"."+tableName)
    if(partitionFlag == "1"):
        sql_list.append("where")
        sql_list.append(partitionCondition)
    output_0 = spark.sql(" ".join(sql_list))
    return [output_0]

def mlp_typeConversion(**params):
    input_0 = params.get("input_0")
    doubleColNameArr = params.get("doubleColName")
    intColNameArr = params.get("intColName")
    stringColNameArr = params.get("stringColName")
    if(doubleColNameArr!=""):
        doubleColNames = doubleColNameArr.split(SPLIT_FLAG)
        for colName in doubleColNames:
            input_0 = input_0.withColumn(colName+"_temp",input_0[colName].cast("double"))\
                .drop(colName).withColumnRenamed(colName+"_temp",colName)
    if (intColNameArr != ""):
        intColNames = intColNameArr.split(SPLIT_FLAG)
        for colName in intColNames:
            input_0 = input_0.withColumn(colName+"_temp",input_0[colName].cast("int")) \
                .drop(colName).withColumnRenamed(colName + "_temp", colName)
    if (stringColNameArr != ""):
        stringColNames = stringColNameArr.split(SPLIT_FLAG)
        for colName in stringColNames:
            input_0 = input_0.withColumn(colName+"_temp",input_0[colName].cast("string")) \
                .drop(colName).withColumnRenamed(colName + "_temp", colName)
    return [input_0]

def mlp_stringIndexer(**params):
    input_0 = params.get("input_0")
    featureCols = params.get("featureCols")
    newCols = params.get("newCols")
    featureCol_list = featureCols.split(SPLIT_FLAG)
    newCol_list = newCols.split(SPLIT_FLAG)
    for i in range(len(featureCol_list)):
        labelIndexer = StringIndexer(inputCol=featureCol_list[i], outputCol=newCol_list[i]).fit(input_0)
        input_0 = labelIndexer.transform(input_0)
    return [input_0]

def mlp_dataSplit(**params):
    input_0 = params.get("input_0")
    splitRatio = float(params.get("splitRatio"))
    ratio_list = [splitRatio,1.0-splitRatio]
    output_0, output_1 = input_0.randomSplit(ratio_list)
    return [output_0, output_1]

def mlp_strataSample(**params):
    input_0 = params.get("input_0")
    strataCol = params.get("strataCol")
    randomSeed = int(params.get("randomSeed"))
    sampleMethod = params.get("sampleMethod")
    sampleRatio = params.get("sampleRatio")
    sampleSize = int(params.get("sampleSize"))
    fractions = {}
    rows = input_0.groupBy(strataCol).count().collect()
    if (sampleMethod == "1"):
        for row in rows:
            if(sampleSize<row['count']):
                fractions[row[strataCol]] = sampleSize/row['count']
            else:
                fractions[row[strataCol]] = 1.0
    else:
        fields = {}
        for field in input_0.schema:
            fields[field.name] = field.dataType
        kvs = sampleRatio.split(";")
        if (isinstance(fields[strataCol], DecimalType)
            or isinstance(fields[strataCol], DoubleType)
            or isinstance(fields[strataCol], FloatType)):
            for kv in kvs:
                values = kv.split(":")
                fractions[float(values[0])] = float(values[1])
        elif (isinstance(fields[strataCol], IntegerType)
              or isinstance(fields[strataCol], LongType)):
            for kv in kvs:
                values = kv.split(":")
                fractions[int(values[0])] = float(values[1])
        else:
            for kv in kvs:
                values = kv.split(":")
                fractions[values[0]] = float(values[1])
    output_0 = input_0.sampleBy(col=strataCol, fractions=fractions,seed=randomSeed)
    return [output_0]

def mlp_decisionTreeClassification(**params):
    input_0 = params.get("input_0")
    featureCols = params.get("featureCols")
    labelCol = params.get("labelCol")
    predictionCol = params.get("predictionCol")
    maxDepth = int(params.get("maxDepth"))
    maxBins = int(params.get("maxBins"))
    minInstancesPerNode = int(params.get("minInstancesPerNode"))
    minInfoGain = float(params.get("minInfoGain"))
    featureCol_list = featureCols.split(SPLIT_FLAG)
    vectorAssembler = VectorAssembler(inputCols=featureCol_list,outputCol=VECTOR_FEATURE)
    classifier = DecisionTreeClassifier(labelCol=labelCol,
                                        featuresCol=VECTOR_FEATURE,
                                        predictionCol=predictionCol,
                                        rawPredictionCol="raw_" + predictionCol,
                                        maxDepth=maxDepth,
                                        maxBins=maxBins,
                                        minInstancesPerNode=minInstancesPerNode,
                                        minInfoGain=minInfoGain)
    pipeline = Pipeline(stages=[ vectorAssembler, classifier])
    model = pipeline.fit(input_0)
    return [model]

def mlp_applyModel(**params):
    input_0 = params.get("input_0")
    input_1 = params.get("input_1")
    output_0 = input_0.transform(input_1)
    return [output_0]

def mlp_binaryClassificationMetrics(**params):
    input_0 = params.get("input_0")
    predictionCol = params.get("predictionCol")
    labelCol = params.get("labelCol")
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="raw_"+predictionCol, labelCol=labelCol)
    areaUnderROC = evaluator.evaluate(input_0)
    areaUnderPR = evaluator.evaluate(input_0, {evaluator.metricName: "areaUnderPR"})
    output_0 = {}
    output_0["areaUnderROC"] = areaUnderROC
    output_0["areaUnderPR"] = areaUnderPR
    return [output_0]

def mlp_exportToHive(**params):
    input_0 = params.get("input_0")
    dbName = params.get("dbName")
    tableName = params.get("tableName")
    tableCols = params.get("tableCols").split(SPLIT_FLAG)
    saveMode = params.get("saveMode")
    saveModeName = ""
    if (saveMode == "2"):
        saveModeName = "overwrite"
    elif (saveMode == "3"):
        saveModeName = "ignore"
    elif (saveMode == "4"):
        saveModeName = "error"
    else:
        saveModeName = "append"
    input_0 = input_0.selectExpr(tableCols)
    input_0.write.mode(saveModeName).insertInto(dbName+"."+tableName)
    return [input_0]

def mlp_histogram(**params):

    return []

def mlp_networkgram(**params):

    return []

if __name__ == "__main__":
    spark = SparkSession.builder.appName("mlp_20190428120626").enableHiveSupport().config("HADOOP_USER_NAME","pto").getOrCreate()
    result_34607 = mlp_dbHive(spark=spark,accountPeriodCols="",accountPeriodType="1",partitionFlag="0",dbName="mlp",accountPeriodDate="",partitionCondition="",tableName="wqf_sample_fa_1",columnName="monthid|latn_id|main_nbr|sub_nbr|oper_type|flag|zj_cnt|bj_cnt|call_cnt|zj_dur|bj_dur|tt_dur|zj_s90_cnt|bj_s90_cnt|zj_l90_cnt|bj_l90_cnt|zj_s60_cnt|bj_s60_cnt|zj_l60_cnt|bj_l60_cnt|zj_s120_cnt|bj_s120_cnt|zj_l120_cnt|bj_l120_cnt|zj_s180_cnt|bj_s180_cnt|zj_l180_cnt|bj_l180_cnt|zj_min_dur|bj_min_dur|zj_max_dur|bj_max_dur|week_am_zj_cnt|week_am_bj_cnt|week_am_tt_cnt|week_am_zj_cnt1|week_am_bj_cnt1|week_am_tt_cnt1|week_am_zj_cnt2|week_am_bj_cnt2|week_am_tt_cnt2|week_pm_zj_cnt|week_pm_bj_cnt|week_pm_tt_cnt|weeked_zj_cnt|weeked_bj_cnt|weeked_tt_cnt|calling_cnt_k|called_cnt_k|call_cnt_k|calling_dur_k|called_dur_k|call_dur_k|zj_s90_cnt_k|bj_s90_cnt_k|zj_l90_cnt_k|bj_l90_cnt_k|zj_s60_cnt_k|bj_s60_cnt_k|zj_l60_cnt_k|bj_l60_cnt_k|zj_s120_cnt_k|bj_s120_cnt_k|zj_l120_cnt_k|bj_l120_cnt_k|zj_s180_cnt_k|bj_s180_cnt_k|zj_l180_cnt_k|bj_l180_cnt_k|zj_min_dur_k|bj_min_dur_k|zj_max_dur_k|bj_max_dur_k|week_am_callingcnt_k|week_am_calledcnt_k|week_am_callcnt_k|week_am_callingcnt_k1|week_am_calledcnt_k1|week_am_callcnt_k1|week_am_callingcnt_k2|week_am_calledcnt_k2|week_am_callcnt_k2|week_pm_callingcnt_k|week_pm_calledcnt_k|week_pm_callcnt_k|weeked_callingcnt_k|weeked_calledcnt_k|weeked_callcnt_k")
    result_34608 = mlp_typeConversion(spark=spark,input_0=result_34607[0],doubleColName="flag",intColName="",stringColName="")
    result_34609 = mlp_stringIndexer(spark=spark,input_0=result_34608[0],newCols="index_oper_type",featureCols="oper_type",derivativeColumns="index_oper_type")
    result_34610 = mlp_dataSplit(spark=spark,input_0=result_34609[0],splitRatio="0.8")
    result_34688 = mlp_strataSample(spark=spark,input_0=result_34610[0],randomSeed="99",strataCol="flag",sampleMethod="2",sampleRatio="0.0:0.1;1.0:1",sampleSize="100")
    result_34611 = mlp_decisionTreeClassification(spark=spark,input_0=result_34688[0],maxDepth="10",maxBins="32",featureCols="zj_cnt|bj_cnt|call_cnt|zj_dur|bj_dur|tt_dur|zj_s90_cnt|bj_s90_cnt|zj_l90_cnt|bj_l90_cnt|zj_s60_cnt|bj_s60_cnt|zj_l60_cnt|bj_l60_cnt|zj_s120_cnt|bj_s120_cnt|zj_l120_cnt|bj_l120_cnt|zj_s180_cnt|bj_s180_cnt|zj_l180_cnt|bj_l180_cnt|zj_min_dur|bj_min_dur|zj_max_dur|bj_max_dur|week_am_zj_cnt|week_am_bj_cnt|week_am_tt_cnt|week_am_zj_cnt1|week_am_bj_cnt1|week_am_tt_cnt1|week_am_zj_cnt2|week_am_bj_cnt2|week_am_tt_cnt2|week_pm_zj_cnt|week_pm_bj_cnt|week_pm_tt_cnt|weeked_zj_cnt|weeked_bj_cnt|weeked_tt_cnt|calling_cnt_k|called_cnt_k|call_cnt_k|calling_dur_k|called_dur_k|call_dur_k|zj_s90_cnt_k|bj_s90_cnt_k|zj_l90_cnt_k|bj_l90_cnt_k|zj_s60_cnt_k|bj_s60_cnt_k|zj_l60_cnt_k|bj_l60_cnt_k|zj_s120_cnt_k|bj_s120_cnt_k|zj_l120_cnt_k|bj_l120_cnt_k|zj_s180_cnt_k|bj_s180_cnt_k|zj_l180_cnt_k|bj_l180_cnt_k|zj_min_dur_k|bj_min_dur_k|zj_max_dur_k|bj_max_dur_k|week_am_callingcnt_k|week_am_calledcnt_k|week_am_callcnt_k|week_am_callingcnt_k1|week_am_calledcnt_k1|week_am_callcnt_k1|week_am_callingcnt_k2|week_am_calledcnt_k2|week_am_callcnt_k2|week_pm_callingcnt_k|week_pm_calledcnt_k|week_pm_callcnt_k|weeked_callingcnt_k|weeked_calledcnt_k|weeked_callcnt_k|index_oper_type",minInfoGain="0.0",derivativeColumns="pre_flag",labelCol="flag",minInstancesPerNode="1",predictionCol="pre_flag")
    result_34638 = mlp_applyModel(spark=spark,input_0=result_34611[0],input_1=result_34610[1])
    result_34639 = mlp_binaryClassificationMetrics(spark=spark,input_0=result_34638[0],positiveLabel="1.0",labelCol="flag",featureCols="zj_cnt|bj_cnt|call_cnt|zj_dur|bj_dur|tt_dur|zj_s90_cnt|bj_s90_cnt|zj_l90_cnt|bj_l90_cnt|zj_s60_cnt|bj_s60_cnt|zj_l60_cnt|bj_l60_cnt|zj_s120_cnt|bj_s120_cnt|zj_l120_cnt|bj_l120_cnt|zj_s180_cnt|bj_s180_cnt|zj_l180_cnt|bj_l180_cnt|zj_min_dur|bj_min_dur|zj_max_dur|bj_max_dur|week_am_zj_cnt|week_am_bj_cnt|week_am_tt_cnt|week_am_zj_cnt1|week_am_bj_cnt1|week_am_tt_cnt1|week_am_zj_cnt2|week_am_bj_cnt2|week_am_tt_cnt2|week_pm_zj_cnt|week_pm_bj_cnt|week_pm_tt_cnt|weeked_zj_cnt|weeked_bj_cnt|weeked_tt_cnt|calling_cnt_k|called_cnt_k|call_cnt_k|calling_dur_k|called_dur_k|call_dur_k|zj_s90_cnt_k|bj_s90_cnt_k|zj_l90_cnt_k|bj_l90_cnt_k|zj_s60_cnt_k|bj_s60_cnt_k|zj_l60_cnt_k|bj_l60_cnt_k|zj_s120_cnt_k|bj_s120_cnt_k|zj_l120_cnt_k|bj_l120_cnt_k|zj_s180_cnt_k|bj_s180_cnt_k|zj_l180_cnt_k|bj_l180_cnt_k|zj_min_dur_k|bj_min_dur_k|zj_max_dur_k|bj_max_dur_k|week_am_callingcnt_k|week_am_calledcnt_k|week_am_callcnt_k|week_am_callingcnt_k1|week_am_calledcnt_k1|week_am_callcnt_k1|week_am_callingcnt_k2|week_am_calledcnt_k2|week_am_callcnt_k2|week_pm_callingcnt_k|week_pm_calledcnt_k|week_pm_callcnt_k|weeked_callingcnt_k|weeked_calledcnt_k|weeked_callcnt_k|index_oper_type",predictionCol="pre_flag")
    result_34695 = mlp_exportToHive(spark=spark,input_0=result_34638[0],tableCols="monthid|latn_id|main_nbr|sub_nbr|oper_type|flag|zj_cnt|bj_cnt|call_cnt|zj_dur|bj_dur|tt_dur|zj_s90_cnt|bj_s90_cnt|zj_l90_cnt|bj_l90_cnt|zj_s60_cnt|bj_s60_cnt|zj_l60_cnt|bj_l60_cnt|zj_s120_cnt|bj_s120_cnt|zj_l120_cnt|bj_l120_cnt|zj_s180_cnt|bj_s180_cnt|zj_l180_cnt|bj_l180_cnt|zj_min_dur|bj_min_dur|zj_max_dur|bj_max_dur|week_am_zj_cnt|week_am_bj_cnt|week_am_tt_cnt|week_am_zj_cnt1|week_am_bj_cnt1|week_am_tt_cnt1|week_am_zj_cnt2|week_am_bj_cnt2|week_am_tt_cnt2|week_pm_zj_cnt|week_pm_bj_cnt|week_pm_tt_cnt|weeked_zj_cnt|weeked_bj_cnt|weeked_tt_cnt|calling_cnt_k|called_cnt_k|call_cnt_k|calling_dur_k|called_dur_k|call_dur_k|zj_s90_cnt_k|bj_s90_cnt_k|zj_l90_cnt_k|bj_l90_cnt_k|zj_s60_cnt_k|bj_s60_cnt_k|zj_l60_cnt_k|bj_l60_cnt_k|zj_s120_cnt_k|bj_s120_cnt_k|zj_l120_cnt_k|bj_l120_cnt_k|zj_s180_cnt_k|bj_s180_cnt_k|zj_l180_cnt_k|bj_l180_cnt_k|zj_min_dur_k|bj_min_dur_k|zj_max_dur_k|bj_max_dur_k|week_am_callingcnt_k|week_am_calledcnt_k|week_am_callcnt_k|week_am_callingcnt_k1|week_am_calledcnt_k1|week_am_callcnt_k1|week_am_callingcnt_k2|week_am_calledcnt_k2|week_am_callcnt_k2|week_pm_callingcnt_k|week_pm_calledcnt_k|week_pm_callcnt_k|weeked_callingcnt_k|weeked_calledcnt_k|weeked_callcnt_k|index_oper_type|pre_flag|probability",saveFileFormat="TEXTFILE",saveMode="2",dbName="mlp",tableFlag="0",tablePartitionCols="",tableName="result_data_stc_001")
    spark.stop()
