import os
import shutil
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans

os.environ["PYSPARK_PYTHON"] = "/home/lmtruong1512/anaconda3/bin/python"


def create_session(memory, warehouse):
    return SparkSession.builder.appName("My app")       \
        .config('spark.driver.memory', memory)          \
        .config('spark.sql.warehouse.dir', warehouse)   \
        .config('spark.rdd.compress', True)            \
        .config('spark.driver.bindAddress', "127.0.0.1") \
        .getOrCreate()


def save_to_parquet(spark, files, batch_size, parquet_name):
    for i in range(0, len(files), batch_size):
        start = i
        end = i + batch_size

        print(f"Loading {start}..{end-1}")
        arrs = [np.load(x, allow_pickle=True) for x in files[start:end]]
        dataset = map(
            lambda x: (Vectors.dense(x), ),
            [x for arr in arrs if arr['arr_0'].ndim == 2 for x in arr["arr_0"]]
        )
        df = spark.createDataFrame(
            dataset, schema=["features"], samplingRatio=1)
        df.write.format('parquet').mode('append').saveAsTable('temporary')
        df.unpersist()

    # Compact files
    warehouse = spark.conf.get('spark.sql.warehouse.dir', 'spark-warehouse')
    tmp_parquet = os.path.join(warehouse, 'temporary')
    df = spark.read.parquet(tmp_parquet)
    df.write.format('parquet').mode('overwrite').saveAsTable(parquet_name)

    # Clean-up
    shutil.rmtree(tmp_parquet)


def convert_parquet(memory, warehouse, path_input, name_path_output):
    spark = create_session(memory, warehouse)
    # print("OK")
    # return
    listdir = os.listdir(path_input)
    listfile = [os.path.join(path_input, filename) for filename in listdir]
    save_to_parquet(spark, listfile, 100, name_path_output)
    print("DONE!!")


memory = "4g"
warehouse = "/home/lmtruong1512/Codes/BTL_CSDLDPT/feature_parquet"
path_input = "/home/lmtruong1512/Codes/BTL_CSDLDPT/extracted_files/extracted_SIFT"
name_path_output = "sift_parquet"
convert_parquet(memory, warehouse, path_input, name_path_output)
