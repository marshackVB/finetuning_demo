# MosaicML deep dive for Databricks technical field

MosaicML makes fine-tuning and pretraining LLMs much easier and more scalable than using open-source libraries on Databricks clusters. This mini-workshop will walk through the process of fine-tuning a foundation model in MosaicML, including the following steps:
1. Set up a development environment to use the Databricks-MosaicML integration
2. Acquire source data for continued pretraining and fine-tuning tasks
3. Use PySpark to process the raw source data into a format amenable to MosaicML
4. Use the MosaicML CLI/SDK to carry out the fine-tuning tasks
5. Review the model training metrics and sample prompts in MLflow, and the training logs in MCLI
6. Provision a model serving endpoint from the custom model that was registered to Unity Catalog
7. Evaluate model performance on new prompts

The demo will adapt an existing MosaicML end-to-end [demo](https://github.com/mosaicml/examples/tree/main/examples/end-to-end-examples/sec_10k_qa) to leverage much of the integrated Databricks-MosaicML stack as of this writing (December 2023).

## Prerequisites
We recommend completing all of the prerequisites before attempting to run any of the notebooks. Any key resources created in the prerequisites and needed in the workshop will be referenced in the `config` notebook. We will identify them as we go.

### MosaicML account
If you don't already have an account, request access to the MosaicML playground & fine-tuning API through go/getfinetuning. More details are available in the [fine-tuning launch email](https://docs.google.com/document/d/1D8z0Y4iRgQOLXQr2VclCQvpzxnu0_nWT5x8iSkXKml8/edit) sent to bricksters. 

### MCLI access
1. Go to the [MosaicML console](https://console.mosaicml.com/account) and create an API key
2. Create Databricks secrets in e2-dogfood. In the `config` notebook, use this secret scope and key to populate the `mcli_secret_scope` and `mcli_api_key` values
3. (Optional, recommended) if you prefer to use a terminal or IDE, follow the [Getting Started docs](https://docs.mosaicml.com/en/latest/getting_started.html) to set up mcli locally

### AWS resources
Follow the guidance in the [Field Eng Cloud Resources Guide](https://databricks.atlassian.net/wiki/spaces/FE/pages/1947665233/Field-eng+Cloud+Resources+Guide#AWS-Cloud-Resources). Because we need to create IAM Roles, we will operate within the aws-sandbox-field-eng AWS account, which provides temporary resources only. Follow the guide's instructions to log into this account and create the following:
1. An S3 bucket (`s3_bucket` config value)
2. Folders in the bucket to hold training data and model checkpoints (`s3_folder_continued_pretrain_train`, `s3_folder_continued_pretrain_validation`, `s3_folder_checkpoints_cpt`, `s3_folder_checkpoints_ift` config values). The values do not need to be updated from the defaults; just make sure that you match the paths of the folders that you actually create.
3. IAM User. This user needs full S3 permissions to the bucket you created
4. AWS Access Key for the IAM User. Record these values; you will need to use them in two places in the demo:
    1. Set up authentication from MosaicML to your S3 bucket by following the steps in the [MosaicML S3 docs](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/s3.html). This is most easily done from your local terminal, where you hopefully set up mcli in the previous step.
    2. Your Databricks cluster will need access to S3, so be sure to add them as secrets to e2-dogfood (the `aws_secret_scope`, `aws_access_key` and `aws_secret_access_key` config values)

### Databricks
MosaicML will need access to a Databricks workspace to log metrics and model checkpoints. 
1. Create a PAT in e2-dogfood
2. Create a [Databricks secret in mcli](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/databricks.html) using this PAT

### Unity Catalog
We recommend creating a UC schema (`uc_schema` config) for this workshop. It will hold the training data tables (`uc_table` prefix config) and the registered, fine-tuned LLM. The fine-tuning API logs the final model checkpoint directly into the UC model registry in the transformers mlflow model flavor, so it can easily be deployed for optimized model serving.

### MLflow 
We recommend creating an MLflow experiment for each combination of training data and model training objective. In this case, that's an experiment for the continued pretraining step (`mlflow_experiment_name_cpt` config) and an experiment for the instruction fine-tuning step (`mlflow_experiment_name_ift` config). The fine-tuning API will log the training run configuration, metrics and (optionally) sample prompt generations directly to MLflow.

### Compute
The heavy lifting of LLM fine-tuning will be handled by MosaicML's serverless compute service, MCloud. On the Databricks side, you just need a modest compute cluster to run data prep workloads, and to interact with the model serving endpoint. For this, a small cluster running MLR 13.3 LTS or later will suffice. Only CPUs are needed. 

When you provision the cluster, be sure to create environment variables in the cluster configs that provide the `boto3` client secure access to the S3 credentials you established previously. 
```
AWS_SECRET_ACCESS_KEY={{secrets/scope/aws_secret_access_key}}
AWS_ACCESS_KEY_ID={{secrets/scope/aws_access_key_id}}
```

## How to run through the workshop
1. Complete all the prerequisites, populating the config notebook as you go
2. Run the `feature_transforms` notebook
3. Run the `finetune` notebook
4. (Future) Run the `deployment` notebook
5. (Optional) review the yaml files for continued pretraining and instruction fine-tuning
6. (Optional) compare sample prompts in the AI playground between the custom model you've deployed and llama2-70b-chat for SEC-related questions
