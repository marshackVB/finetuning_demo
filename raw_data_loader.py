# Databricks notebook source
# MAGIC %pip install datasets -q

# COMMAND ----------

from collections import namedtuple
import numpy as np
import pandas as pd
from datasets import load_dataset
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType, LongType, MapType, DateType
import pyspark.sql.functions as func
from pyspark.sql.functions import col

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

train, validation, test = dataset_to_dataframes()

# COMMAND ----------

display(train)

# COMMAND ----------

#display(sentence_counts_by_doc)

# COMMAND ----------

# MAGIC %md #### Collected sentences into documents

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
                            ((train.doc_id == sentence_counts_by_doc.doc_id) &
                            (train.sentence_count==sentence_counts_by_doc.total_sentences)), 
                            "inner")
                       .drop(sentence_counts_by_doc.doc_id)
                       .withColumn("doc", func.concat_ws(" ", col("doc_array")))
                       .select("doc_id", "doc"))

  return documents

# COMMAND ----------

docs = convert_sentences_to_documents(train)
display(docs)

# COMMAND ----------

docs.count()

# COMMAND ----------

display(train.filter(col("doc_id") == "0000001800_10-K_2007")).orderBy(col("sentence_count").asc())

# COMMAND ----------

train.sort_values(['doc_id', 'sentence_count'], inplace=True)

train[train.doc_id == "0000001800_10-K_2007"].head()

# COMMAND ----------

grouped = train.groupBy(col('doc_id')).agg(func.concat_ws(" ", func.collect_list(col('sentence'))))

# COMMAND ----------

display(grouped)

# COMMAND ----------

f.concat_ws(", ", f.collect_list(df.col2))

# COMMAND ----------



# COMMAND ----------

sample_doc = train.filter(col("doc_id") == "0000001750_10-K_2020").toPandas()

# COMMAND ----------

sample_doc_lst = sample_doc[['doc_id', 'sentence_count', 'sentence']].sort_values(['doc_id', 'sentence_count']).values.tolist()

sample_doc_list = [(doc_id, sentence_count, sentence) for 
                    doc_id, sentence_count, sentence in sample_doc_lst]

# COMMAND ----------

sample_doc_list[0]

# COMMAND ----------

for doc_id, sentence_num, sentence in sample_doc_list:
  print(doc_id, sentence_num, get_token_lesentence)

# COMMAND ----------

x = [[1,2,3]]
y = [[4]]

print(x.append(y))


# COMMAND ----------

def get_token_len(text):
  token_len = len(text.split())
  return token_len

token_limit = 25
batch = ""
batches = []
prepend_to_next_batch = ""

for doc_id, sentence_num, sentence in sample_doc_list:
  # If some tokens from prior sentence could not fit in
  # in prior batch, append to current sentence

  # If the batch is full, add it to the batches and reset the batch
  sentence = prepend_to_next_batch + sentence
  sentence_length = get_token_len(sentence)
  batch_length = get_token_len(batch)
  print(f"""
        sentence_length: {sentence_length}, 
        prepend_to_next_batch: {get_token_len(prepend_to_next_batch)}""")

  # Count remaining tokens within limit
  tokens_remaining_in_batch = token_limit - batch_length
  # Filter tokens within limit"
  sentence_tokens = sentence.split()
  append_to_batch = sentence_tokens[:tokens_remaining_in_batch]
  # Filter tokens outside of limit"
  prepend_to_next_batch = sentence_tokens[tokens_remaining_in_batch:]
  # Append tokens within limit to batch
  batch = batch + " " + (" ".join(append_to_batch))
  # Prepend tokens outside limit to next batch
  prepend_to_next_batch = " ".join(prepend_to_next_batch) + " "

  if batch_length == token_limit:
    # If batch is full, add batch to batches and reset batch
    batches = batches + ([[batch]])
    batch = ""

# COMMAND ----------

batches[0][0]

# COMMAND ----------

batches[1][0]

# COMMAND ----------

batches[2][0]

# COMMAND ----------

batches[3][0]

# COMMAND ----------

len(batch.split())

# COMMAND ----------

def get_token_len(text):
  token_len = len(text.split())
  return token_len

token_limit = 25
batch = ""
batches = []
prepend_to_next_batch = ""

for doc_id, sentence_num, sentence in sample_doc_list:
  # If some tokens from prior sentence could not fit in
  # in prior batch, append to current sentence
  sentence = prepend_to_next_batch + sentence
  sentence_length = get_token_len(sentence)
  # Count remaining tokens within limit
  tokens_remaining_in_batch = token_limit - batch_length
  # Filter tokens within limit"
  sentence_tokens = sentence.split()
  append_to_batch = sentence_tokens[:tokens_remaining_in_batch]
  # Filter tokens outside of limit"
  prepend_to_next_batch = sentence_tokens[tokens_remaining_in_batch:]
  # Append tokens within limit to batch
  batch = batch + " " + (" ".join(append_to_batch))
  # Prepend tokens outside limit to next batch
  prepend_to_next_batch = " ".join(prepend_to_next_batch) + " "

  if get_token_len(batch) == token_limit:
    # If batch is full, add batch to batches and reset batch
    batches = batches + ([[batch]])
    batch = ""
    prepend_to_next_batch = ""

# COMMAND ----------

batches

# COMMAND ----------

def make_convert_text_to_tokens(model_name, bos_text, eos_text, MAX_LENGTH=2048, is_pandas_udf=False):
  def as_pandas_udf(texts):
    return apply(texts)
  
  def apply(texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bos_tokens = tokenizer(bos_text,truncation=False,padding=False, add_special_tokens=False)['input_ids']
    eos_tokens = tokenizer(eos_text,truncation=False,padding=False, add_special_tokens=False)['input_ids']

    for batch in texts:
      res = []
      for text in batch:
        tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]
        buf = bos_tokens + tokens + eos_tokens
        N = len(buf)
        s = 0
        e = min(N, MAX_LENGTH)
        parts = []
        while s < N:
          parts.append( np.asarray( buf[s:e]).tobytes())

          s = e
          e = min(N, e + MAX_LENGTH)
        res = res + [parts]
      yield pd.Series(res)
  
  if not is_pandas_udf:
    return apply
  else:
    return as_pandas_udf

# COMMAND ----------

sample_doc = train.filter(col("doc_id") == "0000001750_10-K_2020").toPandas()

# COMMAND ----------

import datasets

# COMMAND ----------

hf_dataset = datasets.Dataset.from_pandas(df=sample_doc, split="train")

# COMMAND ----------

for sample in hf_dataset:
  print(sample)

# COMMAND ----------

hf_dataset = hf_datasets.Dataset.from_pandas(df=df, split=args['split'])
tokenizer = AutoTokenizer.from_pretrained(args['tokenizer'])
# we will enforce length, so suppress warnings about sequences too long for the model
tokenizer.model_max_length = int(1e30)
max_length = args['concat_tokens']

for sample in hf_dataset:

    buffer = []
    for sample in hf_dataset:
        encoded = tokenizer(sample['words'],
                            truncation=False,
                            padding=False)
        iids = encoded['input_ids']
        buffer = buffer + iids
        while len(buffer) >= max_length:
            concat_sample = buffer[:max_length]
            buffer = []
            yield {
                # convert to bytes to store in MDS binary format
                'tokens': np.asarray(concat_sample).tobytes()
            }

# COMMAND ----------

for sample in hf_dataset:
