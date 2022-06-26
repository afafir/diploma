from itertools import chain

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql import functions as f, DataFrame
from pyspark.sql.functions import  create_map, lit

from src.dependencies.model import attack_dict


class LabelBinaryConverter(Transformer):

    @keyword_only
    def __init__(self):
        super(LabelBinaryConverter, self).__init__()

    def _transform(self, df: DataFrame):
        return df.withColumn('labelBinary', f.when(f.col('labels') == 'normal', 'normal')
                             .otherwise('attack'))


class LabelMulticlassConverter(Transformer):

    @keyword_only
    def __init__(self):
        super(LabelMulticlassConverter, self).__init__()

    def _transform(self, df: DataFrame):
        mapping_expr = create_map([lit(x) for x in chain(*attack_dict.items())])
        return df.withColumn('labelMulti', mapping_expr[df['labels']])
