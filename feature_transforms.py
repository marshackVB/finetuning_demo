# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature transformations for continued pretraining
# MAGIC   1. Download datasets from the huggingface datasets hub
# MAGIC   2. Combind sentences into documents
# MAGIC   3. Write documents to S3

# COMMAND ----------

# MAGIC %pip install datasets composer -q

# COMMAND ----------

from collections import namedtuple
from tempfile import TemporaryDirectory
from pathlib import Path
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from pyspark.sql.types import StructType, StringType
import pyspark.sql.functions as func
from pyspark.sql.types import (StructType, 
                               StructField, 
                               StringType, 
                               IntegerType, 
                               MapType)
                               
from pyspark.sql.functions import col
from composer.utils import ObjectStore, maybe_create_object_store_from_uri

# COMMAND ----------

# MAGIC %md
# MAGIC Download pre-training data from the Huggingface datasets hub

# COMMAND ----------

def dataset_to_dataframes(data_subset="small_lite"):
  """
  Given a transformers datasets name, download the dataset and 
  return Spark DataFrame versions. Result include train and test
  dataframes as well as a dataframe of label index to string
  representation.
  """

  spark_datasets = namedtuple("spark_datasets", "train validation test")
  
  # Define Spark schemas
  schema = StructType([StructField("cik", StringType(), False),
                       StructField("sentence", StringType(), False),
                       StructField("section", IntegerType(), False),
                       StructField("labels", MapType(StringType(), IntegerType()), False),
                       StructField("filing_date_str", StringType(), False),
                       StructField("doc_id", StringType(), False),
                       StructField("sentence_id", StringType(), False),
                       StructField("sentence_count", IntegerType(), False)])

  dataset_name = 'JanosAudran/financial-reports-sec'
  dataset = load_dataset(dataset_name,
                         data_subset)
  
  train_pd  = dataset['train'].to_pandas()
  validation_pd  = dataset['validation'].to_pandas()
  test_pd  =  dataset['test'].to_pandas()

  def to_spark_df(pandas_df):
    """
    Convert a Pandas DataFrame to a SparkDataFrame and convert the date
    columns from a string format to a date format.
    """
    spark_df = (spark.createDataFrame(pandas_df, schema=schema)
                     .withColumn("filing_date", func.to_date(col("filing_date_str"), "yyyy-MM-dd"))
                     .drop(col("filing_date_str")))
    return spark_df
  
  train = to_spark_df(train_pd)
  validation = to_spark_df(validation_pd)
  test = to_spark_df(test_pd)

  return spark_datasets(train, validation, test)

# COMMAND ----------

train_sentences, validation_sentences, test_sentences = dataset_to_dataframes()

# COMMAND ----------

# MAGIC %md
# MAGIC Data is at a sentence level where sentences are ordered ('sentence_count') to create a document ('document_id'). Document correspond to different companies (identified by unique 'cik')

# COMMAND ----------

display(train_sentences)

# COMMAND ----------

# MAGIC %md
# MAGIC Group, order and concatenate sentences into documents.

# COMMAND ----------

def convert_sentences_to_documents(spark_df):

  from pyspark.sql.window import Window 

  sentence_counts_by_doc = (spark_df.groupby(col("doc_id"))
                                    .agg(func.max(col("sentence_count"))
                                    .alias('total_sentences')))

  documents = (spark_df.withColumn("doc_array",
                                func.collect_list(col("sentence")).over(
                                Window.partitionBy("doc_id")
                                      .orderBy(col("sentence_count").asc())))
                       .join(sentence_counts_by_doc, 
                            ((spark_df.doc_id == sentence_counts_by_doc.doc_id) &
                             (spark_df.sentence_count==sentence_counts_by_doc.total_sentences)), 
                            "inner")
                       .drop(sentence_counts_by_doc.doc_id)
                       .withColumn("doc", func.concat_ws(" ", col("doc_array")))
                       .select("doc_id", "doc"))

  return documents

# COMMAND ----------

train_docs = convert_sentences_to_documents(train_sentences)
validation_docs = convert_sentences_to_documents(validation_sentences)
display(train_docs)

# COMMAND ----------

train_docs_cnt = train_docs.count()
validation_docs_cnt = validation_docs.count()

print(f"Training docs: {train_docs_cnt}, Validation docs: {validation_docs_cnt}")

# COMMAND ----------

# MAGIC %md
# MAGIC Write training and evalution documents to S3

# COMMAND ----------

def config_s3_writer(id_col, text_col, s3_bucket, s3_folder):

  def write_to_s3(pandas_df):

    pd.set_option('display.max_colwidth', None)

    document_id = pandas_df[id_col].to_string(header=False, index=False)
    text_to_dump = pandas_df[text_col].to_string(header=False, index=False)
    object_store = maybe_create_object_store_from_uri(s3_bucket)

    with TemporaryDirectory() as _tmp_dir:
      ticker_dir = Path(document_id)
      ticker_dir_full = _tmp_dir / ticker_dir
      os.makedirs(ticker_dir_full, exist_ok=True)

      text_file_name = f'{document_id}.txt'

      local_text_file_path = Path(ticker_dir_full) / Path(text_file_name)
      with open(local_text_file_path, 'w') as _txt_file:
        _txt_file.write(text_to_dump)

      object_name = f"{s3_folder}/{text_file_name}"
    
      object_store.upload_object(object_name=object_name,
                                 filename=str(local_text_file_path))
      
    return pd.DataFrame([object_name], columns=['file_path'])
    
  return write_to_s3

# COMMAND ----------

spark_schema = StructType()
spark_schema.add("file_path", StringType())

data_params = [(train_docs_cnt, "data/train", train_docs),
               (validation_docs_cnt, "data/validation", validation_docs)]

for doc_cnt, save_folder, spark_df in data_params:
  
  s3_writer_udf = config_s3_writer(id_col="doc_id", 
                                  text_col="doc", 
                                  s3_bucket="s3://mosaicml-demo/", 
                                  s3_folder=save_folder)
                                  
  s3_training_files = (spark_df.repartition(doc_cnt, "doc_id")
                               .groupBy('doc_id')
                               .applyInPandas(s3_writer_udf, schema=spark_schema))

  s3_training_files.write.mode('overwrite').format('delta').saveAsTable(f"default.mlc_mosaic_file_paths_{save_folder.split('/')[0]}")
