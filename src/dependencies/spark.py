import json
from os import listdir, path

from pyspark import SparkFiles
from pyspark.sql import SparkSession

def start(app_name='my_app', jar_packages=[], files=[], spark_config={}):
    spark_builder = (
        SparkSession
        .builder
        .appName(app_name)
    )

    spark_sess = spark_builder.getOrCreate()
    spark_sess.sparkContext.setLogLevel('WARN')

    spark_files_dir = SparkFiles.getRootDirectory()
    config_files = [filename
                    for filename in listdir(spark_files_dir)
                    if filename.endswith('config.json')]

    if config_files:
        path_to_config_file = path.join(spark_files_dir, config_files[0])
        with open(path_to_config_file, 'r') as config_file:
            config_dict = json.load(config_file)
    else:
        config_dict = None

    return spark_sess, config_dict

