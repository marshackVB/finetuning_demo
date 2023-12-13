# Databricks notebook source
# MAGIC %md
# MAGIC **Credentials**. 
# MAGIC
# MAGIC  - [Databricks credentials](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/databricks.html)
# MAGIC    - Optional credentials check
# MAGIC  - [AWS credentials](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/s3.html)
# MAGIC    - Access key and secret access key should be a their own file under [default] 
# MAGIC  - MosaicML credentials as [secret](https://docs.databricks.com/en/security/secrets/secrets.html) within [scope](https://docs.databricks.com/en/security/secrets/secret-scopes.html)
# MAGIC   ```
# MAGIC   databricks secrets create-scope mlc_mosaic_scope --profile DOGFOOD
# MAGIC   databricks secrets put-secret mlc_mosaic_scope api_key --profile DOGFOOD
# MAGIC   ```

# COMMAND ----------

# MAGIC %pip install -q mosaicml-cli

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mcli
from mcli import finetune

# COMMAND ----------

mcli.set_api_key(
  dbutils.secrets.get(scope="mlc_mosaic_scope", key="api_key")
                 )

# COMMAND ----------

run = finetune(
    model="mosaicml/mpt-7b",
    train_data_path="mosaicml/instruct-v3/train",
    eval_data_path="mosaicml/instruct-v3/test",
    save_folder="s3://mosaicml-demo/models",
    training_duration="10ba",
    experiment_trackers=[{
         'integration_type': 'mlflow',
         'experiment_name': '/Users/marshall.carter@databricks.com/finetune_experiment',
         'model_registry_prefix': 'main.timl_mosaic'
      }],
)

# COMMAND ----------

import os
import requests

# COMMAND ----------

query = "what is the SEC10K?"
temperature = 0.7
max_tokens = 50
data = {"dataframe_split": {"columns": ["prompt", "temperature", "max_tokens"],
                            "data": [[ query, temperature, max_tokens]]}
        }
headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}

response = requests.post(
    url='https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/mlc_test_model/invocations', json=data, headers=headers
)
#output =response.json()["predictions"][0]["candidates"][0]["text"]
output =response.json()
#print(output)

# COMMAND ----------

os.environ.get("DATABRICKS_TOKEN")

# COMMAND ----------

API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import requests
import json

def query_endpoint(prompt:str) -> dict:
    data = {
        "inputs": {
            "prompt": [prompt]
        },
        "params": {
            "max_tokens": 200, 
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

    # print(json.dumps(response.json()))

    return response.json()

# COMMAND ----------

prompt1 = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the SEC?\n\n### Response:\n"

results1 = query_endpoint(prompt1)
print(results1["predictions"][0]["candidates"][0]["text"])
