import random
from time import time
import time

#import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType, StructType, StructField, IntegerType
#from elasticsearch import Elasticsearch

from src.dependencies.model import columns, categorical_columns
from src.dependencies.spark import start
import telebot

from src.dependencies.transofrmers.LabelTransformers import LabelBinaryConverter, LabelMulticlassConverter

bot = telebot.TeleBot("", parse_mode=None)
attack_form = "ALARM! ALARM! ALARM!\nAttack on network: \n protocol_type: {} \n service: {} \n flag: {}"


#from src.dependencies.transofrmers.LabelTransformers import LabelBinaryConverter, LabelMulticlassConverter


def main():
    spark, config = start(
        app_name='ml_job',
        files=['config/etl_config.json'])

    model = CrossValidatorModel.load("C:/Users/afafi/PycharmProjects/ml_diploma/model")
    train_df = define_dataframe(extract_data(spark, config['test_path'], 32)).limit(1000)
    label_pipeline = Pipeline(stages=[LabelBinaryConverter(),
                                       LabelMulticlassConverter(),
                                       StringIndexer(inputCol="labelBinary", outputCol="labelBinary_index"),
                                       StringIndexer(inputCol="labelMulti", outputCol="labelMulti_index")])
    label_pipeline = label_pipeline.fit(train_df)

    train_df = label_pipeline.transform(train_df)
    train_df, ohe_cols = index_strings_with_ohe(train_df, categorical_columns)
    func_udf = udf(func, StringType())
    cols = ("labels", "labelBinary")
    df = model.transform(train_df)
    df = df.filter(df['predicted'] == 'attack')
    result = train_df.res.collect()
    for x in result:
        bot.send_message("370347545", attack_form.format(x['protocol_type'], x['service'], x['flag']))
    time.sleep(100)


def func():
    if random.random() < 0.75:
        return 'normal'
    else: return 'attack'


def define_dataframe(rdd):
    df = (rdd.toDF(columns)).select(
        col('duration').cast(DoubleType()),
        col('protocol_type').cast(StringType()),
        col('service').cast(StringType()),
        col('flag').cast(StringType()),
        col('src_bytes').cast(DoubleType()),
        col('dst_bytes').cast(DoubleType()),
        col('land').cast(DoubleType()),
        col('wrong_fragment').cast(DoubleType()),
        col('urgent').cast(DoubleType()),
        col('hot').cast(DoubleType()),
        col('num_failed_logins').cast(DoubleType()),
        col('logged_in').cast(DoubleType()),
        col('num_compromised').cast(DoubleType()),
        col('root_shell').cast(DoubleType()),
        col('su_attempted').cast(DoubleType()),
        col('num_root').cast(DoubleType()),
        col('num_file_creations').cast(DoubleType()),
        col('num_shells').cast(DoubleType()),
        col('num_access_files').cast(DoubleType()),
        col('num_outbound_cmds').cast(DoubleType()),
        col('is_host_login').cast(DoubleType()),
        col('is_guest_login').cast(DoubleType()),
        col('count').cast(DoubleType()),
        col('srv_count').cast(DoubleType()),
        col('serror_rate').cast(DoubleType()),
        col('srv_serror_rate').cast(DoubleType()),
        col('rerror_rate').cast(DoubleType()),
        col('srv_rerror_rate').cast(DoubleType()),
        col('same_srv_rate').cast(DoubleType()),
        col('diff_srv_rate').cast(DoubleType()),
        col('srv_diff_host_rate').cast(DoubleType()),
        col('dst_host_count').cast(DoubleType()),
        col('dst_host_srv_count').cast(DoubleType()),
        col('dst_host_same_srv_rate').cast(DoubleType()),
        col('dst_host_diff_srv_rate').cast(DoubleType()),
        col('dst_host_same_src_port_rate').cast(DoubleType()),
        col('dst_host_srv_diff_host_rate').cast(DoubleType()),
        col('dst_host_serror_rate').cast(DoubleType()),
        col('dst_host_srv_serror_rate').cast(DoubleType()),
        col('dst_host_rerror_rate').cast(DoubleType()),
        col('dst_host_srv_rerror_rate').cast(DoubleType()),
        col('labels').cast(StringType())
    )
    return df


def extract_data(spark, path, partition):
    dataset_rdd = spark \
        .sparkContext \
        .textFile(path, partition) \
        .map(lambda line: line.split(','))
    return dataset_rdd


def index_strings_with_ohe(df, labels):
    t0 = time()
    print("Start OHE process")
    ohe_cols = []
    for label in labels:
        distinct_values = df.select(label).distinct().rdd.flatMap(lambda x: x).collect()
        for distinct_value in distinct_values:
            pivot_udf = udf(lambda item: 1 if item == distinct_value else 0, IntegerType())
            column_name = label + "_" + distinct_value
            ohe_cols.append(column_name)
            df = df.withColumn(column_name, pivot_udf(col(label)))
    print(f"End OHE process: {time() - t0}")
    return df, ohe_cols


if __name__ == '__main__':
    main()
