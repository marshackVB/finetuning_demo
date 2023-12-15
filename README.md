# MosaicML deep dive for Databricks technical field

MosaicML makes fine-tuning and pretraining LLMs much easier and more scalable than using open-source libraries on Databricks clusters. This mini-workshop will walk through the process of fine-tuning a foundation model in MosaicML, including the following steps:
1. Set up a development environment to use the Databricks-MosaicML integration
2. Acquire source data for continued pretraining and fine-tuning tasks
3. Use PySpark to process the raw source data into a format amenable to MosaicML
4. Use the MosaicML CLI/SKD to carry out the fine-tuning tasks
5. Review the model training metrics and sample prompts in MLflow, and the training logs in MCLI
6. Provision a model serving endpoint from the custom model that was registered to Unity Catalog
7. Evaluate model performance on new prompts

## Prerequisites

### MosaicML account
If you don't already have an account, request access to the MosaicML playground & fine-tuning API through go/getfinetuning. More details are available in the [fine-tuning launch email](https://docs.google.com/document/d/1D8z0Y4iRgQOLXQr2VclCQvpzxnu0_nWT5x8iSkXKml8/edit) sent to bricksters. 

### AWS resources
Follow the guidance in the [Field Eng Cloud Resources Guide](https://databricks.atlassian.net/wiki/spaces/FE/pages/1947665233/Field-eng+Cloud+Resources+Guide#AWS-Cloud-Resources). Because we need to create IAM Roles, we will operate within the aws-sandbox-field-eng AWS account, which provides temporary resources only. Follow the guide's instructions to log into this account and create the following:
1. An S3 bucket

### MCLI access
1. Go to the [MosaicML console](https://console.mosaicml.com/account) and create an API key
2. If using a 

## How to run through the workshop