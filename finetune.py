# Databricks notebook source
# MAGIC %md
# MAGIC ## Train models with the [Finetuning API](https://docs.mosaicml.com/projects/mcli/en/latest/finetuning/finetuning.html)
# MAGIC 1. Execute a continued pretraining (domain adaptation) job.
# MAGIC 2. Load the pretrained model and execute an instruction fintuning job.
# MAGIC 3. Serve the instruction finetuned model.

# COMMAND ----------

# MAGIC %pip install -q mosaicml-cli

# COMMAND ----------

import os
import requests
import json
import mcli
from mcli import finetune

# COMMAND ----------

mcli.set_api_key(
  dbutils.secrets.get(scope="mlc_mosaic_scope", key="api_key")
                 )

# COMMAND ----------

continued_pretraining = finetune(
        model="mosaicml/mpt-7b",
        task_type='CONTINUED_PRETRAIN',
        train_data_path="s3://mosaicml-demo/data/train",
        eval_data_path="s3://mosaicml-demo/data/validation",
        save_folder="s3://mosaicml-demo/checkpoints",
        training_duration="3ep",
        experiment_trackers=[{
            'integration_type': 'mlflow',
            'experiment_name': '/Users/marshall.carter@databricks.com/finetune_experiment'
        }],
)

# COMMAND ----------

instruction_finetune = finetune(
    model="mosaicml/mpt-7b",
    custom_weights_path="s3://mosaicml-demo/checkpoints/contd-pretrain-mpt-7b-onguvl/checkpoints/ep3-ba102-rank0.pt",
    task_type='INSTRUCTION_FINETUNE',
    train_data_path="mosaicml/instruct-v3/train",
    eval_data_path="mosaicml/instruct-v3/test",
    save_folder="s3://mosaicml-demo/checkpoints",
    training_duration="3ep",
    experiment_trackers=[{
         'integration_type': 'mlflow',
         'experiment_name': '/Users/marshall.carter@databricks.com/finetune_experiment',
         'model_registry_prefix': 'main.timl_mosaic'
      }],
)

# COMMAND ----------

# MAGIC %md Manually deploy model from Unity Catalog and query endpoint

# COMMAND ----------

API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

def query_endpoint(prompt:str) -> dict:
    data = {
        "inputs": {
            "prompt": [prompt]
        },
        "params": {
            "max_tokens": 500, 
            "temperature": 0.0
        }
    }
    headers = {
        "Context-Type": "text/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    response = requests.post(
        url="https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/mlc_test_model/invocations",
        json=data,
        headers=headers
    )

    return response.json()

# COMMAND ----------

question = "What are Abbott Laboratories key company initiatives?"

prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

answer = query_endpoint(prompt)
print(answer["predictions"][0]["candidates"][0]["text"])
