# Databricks notebook source
# MAGIC %md
# MAGIC # Train models with the [Finetuning API](https://docs.mosaicml.com/projects/mcli/en/latest/finetuning/finetuning.html)
# MAGIC 1. Execute a continued pretraining (domain adaptation) job.
# MAGIC 2. Load the pretrained model and execute an instruction fintuning job.
# MAGIC 3. Serve the instruction finetuned model.

# COMMAND ----------

# MAGIC %md ## Library installs & imports, set up MCLI authentication

# COMMAND ----------

# MAGIC %pip install -q mosaicml-cli

# COMMAND ----------

import os
import requests
import json
import mcli

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

mcli.set_api_key(
  dbutils.secrets.get(scope=config["mcli_secret_scope"], key=config["mcli_api_key"])
                 )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Continued pretraining
# MAGIC
# MAGIC Take a pretrained model and provide it with domain-specific language understanding

# COMMAND ----------

continued_pretraining = mcli.create_finetuning_run(
        model="mosaicml/mpt-7b",
        task_type='CONTINUED_PRETRAIN',
        train_data_path=os.path.join(config["s3_bucket"], config["s3_folder_continued_pretrain_train"]),
        eval_data_path=os.path.join(config["s3_bucket"], config["s3_folder_continued_pretrain_validation"]),
        save_folder=os.path.join(config["s3_bucket"], config["s3_folder_checkpoints_cpt"]),
        training_duration=config['cpt_duration'],
        experiment_tracker={
            "mlflow": {
                "experiment_path": config["mlflow_experiment_name_cpt"],
                "model_registry_path": config['uc_schema']
            }
        }
)

# COMMAND ----------

# MAGIC %md
# MAGIC Check on run status. We can use both the fine-tuning specific run status and the general run status calls

# COMMAND ----------

mcli.get_finetuning_runs()

# COMMAND ----------

mcli.get_run(continued_pretraining.name)

# COMMAND ----------

run_info = mcli.get_run(continued_pretraining.name)
run_info.submitted_config

# COMMAND ----------

cpt_checkpoint_path = os.path.join(config["s3_bucket"], config["s3_folder_checkpoints_cpt"], continued_pretraining.name, "checkpoints/ep3-ba102-rank0.pt")
cpt_checkpoint_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tuning
# MAGIC
# MAGIC Equip model with stronger instruction-following capabilities
# MAGIC
# MAGIC In this example, we use the `mosaicml/instruct-v3` dataset from [Huggingface](https://huggingface.co/datasets/mosaicml/instruct-v3). This is a high-quality instruction dataset covering many different task types. 
# MAGIC
# MAGIC It is common to use similar datasets on customers' pretrained models where they don't have their own custom instruction datasets. However, it might make more sense to use a custom fine-tuning dataset, and these can be fed to the fine-tuning API directly from a [Unity Catalog Volume](https://docs.mosaicml.com/projects/mcli/en/latest/finetuning/finetuning.html#supported-data-sources).

# COMMAND ----------

instruction_finetune = mcli.create_finetuning_run(
    model="mosaicml/mpt-7b",
    custom_weights_path=cpt_checkpoint_path,
    task_type='INSTRUCTION_FINETUNE',
    train_data_path="mosaicml/instruct-v3/train",
    eval_data_path="mosaicml/instruct-v3/test",
    save_folder=os.path.join(config["s3_bucket"], config["s3_folder_checkpoints_ift"]),
    training_duration=config['ift_duration'],
    experiment_tracker={
        "mlflow": {
            "experiment_path": config["mlflow_experiment_name_ift"],
            "model_registry_path": config['uc_schema']
        }
    }
)

# COMMAND ----------

mcli.get_run(instruction_finetune.name)

# COMMAND ----------

# MAGIC %md ## Manually deploy model from Unity Catalog and query endpoint

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
        url=f"https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/mlc_test_model/invocations",
        json=data,
        headers=headers
    )

    return response.json()

# COMMAND ----------

question = "What are Abbott Laboratories key company initiatives?"

prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n"

answer = query_endpoint(prompt)
print(answer["predictions"][0]["candidates"][0]["text"])
