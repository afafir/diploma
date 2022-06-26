from time import time

import pyspark.sql.functions as F
import sys
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, \
    LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType, IntegerType

from src.dependencies.model import columns, indexed_columns, categorical_columns, binary_columns, numerical_columns
from src.dependencies.spark import start
from src.dependencies.transofrmers.LabelTransformers import LabelBinaryConverter, LabelMulticlassConverter


def main():
    model = sys.argv[1]
    classification = 'labelBinary_index' if sys.argv[2] == 'binary' else 'labelMulti_index'

    spark, config = start(
        app_name='ml_job',
        files=['config/etl_config.json']
    )
    test_data = define_dataframe(extract_data(spark, config['test_path'], 16))
    training_data = define_dataframe(extract_data(spark, config['training_path'], 16))

    label_pipeline = Pipeline(stages=[LabelBinaryConverter(),
                                      LabelMulticlassConverter(),
                                      StringIndexer(inputCol="labelBinary", outputCol="labelBinary_index"),
                                      StringIndexer(inputCol="labelMulti", outputCol="labelMulti_index")])
    label_pipeline = label_pipeline.fit(training_data)

    test_data = label_pipeline.transform(test_data)
    training_data = label_pipeline.transform(training_data)

    training_data, ohe_cols = index_strings_with_ohe(training_data, categorical_columns)
    test_data, ohe_cols = index_strings_with_ohe(test_data, categorical_columns)

    training_data = training_data.cache()
    test_data = test_data.cache()

    va = VectorAssembler(inputCols=ohe_cols + binary_columns + numerical_columns,
                         outputCol='features')
    training_data = va.transform(training_data) \
        .select("features", col("labelBinary_index").alias("label"))
    test_data = va.transform(test_data) \
        .select("features", col("labelBinary_index").alias("label"))



    evaluator = BinaryClassificationEvaluator(labelCol="label")
    if model == "dt":
        print("START DT")
        model, time0 = decision_tree(training_data, test_data, "label")
        print(f"DT time = {time0}")
    if model == "svm":
        print("START SVM")
        model, time0 = svm(training_data, test_data, "label")
        print(f"SVM time = {time0}")
    if model == "rf":
        print("START RF")
        model, data, time0 = random_forest(training_data, test_data, "label")

        print(f"RF time = {time0}")
    if model == "gbt":
        print("START GBT")
        model, time0 = gradient_boosting(training_data, test_data, "label")
        print(f"GBT time = {time0}")
    model.save("C:/Users/afafi/PycharmProjects/ml_diploma/model")

    #print(f"accuracy: {evaluator.evaluate(model)}")


def feature_selection_chi_square(training_data):
    va = VectorAssembler(inputCols=indexed_columns, outputCol='features')
    training_data = va.transform(training_data) \
        .select("features", "label")
    result = ChiSquareTest.test(training_data, 'features', 'label')
    print("pvalues: " + str(result.select('pValues').collect()))


def extract_data(spark, path, partition):
    dataset_rdd = spark \
        .sparkContext \
        .textFile(path, partition) \
        .map(lambda line: line.split(','))
    return dataset_rdd


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


def index_strings(df, labels):
    ohe_cols = []
    for label in labels:
        unique_values = df.select(label).distinct().rdd.flatMap(lambda x: x).collect()
        for value in unique_values:
            ohe_cols.append(label + "_" + value)
        expr = [F.when(F.col(label) == value, 1).otherwise(0).alias(label + "_" + value) for value in unique_values]
        df = df.select("*", *expr)
    return df.drop(*labels), ohe_cols


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
    return df.drop(*labels), ohe_cols


def svm(train_df, test_df, label):
    t0 = time()
    lr = LinearSVC(labelCol="label", regParam=0.1)
    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01, 0.05]) \
        .addGrid(lr.maxIter, [10, 20, 30, 40, 50]) \
        .addGrid(lr.elasticNetParam, [0.2, 0.5, 0.8]) \
        .build()
    cross_validator = CrossValidator(estimator=lr,
                                     numFolds=3,
                                     estimatorParamMaps=param_grid,
                                     evaluator=MulticlassClassificationEvaluator())
    lsvc = cross_validator.fit(train_df)
    print(f"SVM: {time() - t0}")
    return lsvc.transform(test_df), time() - t0


def decision_tree(train_df, test_df, label):
    t0 = time()
    dt = DecisionTreeClassifier(labelCol=label, featuresCol='features', maxDepth=2, maxBins=80)
    dt_param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [2, 5, 10, 20, 30]) \
        .addGrid(dt.maxBins, [10, 20, 30, 40, 50]) \
        .addGrid(dt.criterion, ['gini', 'entropy']) \
        .build()
    dtcv = CrossValidator(estimator=dt,
                          estimatorParamMaps=dt_param_grid,
                          evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction"),
                          numFolds=5)
    dtcv_model = dtcv.fit(train_df)
    print(f"DECISION TREE TIME: {time() - t0}")
    return dtcv_model.transform(test_df), time() - t0


def random_forest(train_df, test_df, label):
    t0 = time()
    rf = RandomForestClassifier(featuresCol='features', labelCol=label, maxBins=70)
    rf_param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 30, 40]) \
        .build()
    rfcv = CrossValidator(estimator=rf,
                          estimatorParamMaps=rf_param_grid,
                          evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction"),
                          numFolds=5)
    model = rfcv.fit(train_df)
    return model, model.transform(test_df), time() - t0


def gradient_boosting(train_df, test_df, label):
    t0 = time()
    gbt = GBTClassifier(labelCol=label, featuresCol="features", maxIter=10)
    gbt_param_grid = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [5, 10, 15]) \
        .addGrid(gbt.maxDepth, [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]) \
        .addGrid(gbt.maxBins, [70, 80, 90, 100, 120]) \
        .build()
    gbt_cv = CrossValidator(estimator=gbt,
                            estimatorParamMaps=gbt_param_grid,
                            evaluator=BinaryClassificationEvaluator(rawPredictionCol='rawPrediction'),
                            numFolds=5)
    model = gbt_cv.fit(train_df)
    return model.transform(test_df), time()-t0



if __name__ == '__main__':
    main()
