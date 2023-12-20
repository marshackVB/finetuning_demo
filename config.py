# Databricks notebook source
config = {
    "max_length": 2048,
    "eval_prompts": [],
    "s3_bucket": "s3://<your-bucket-name>",
    "s3_folder_continued_pretrain_train": "sec-10k/data/train",
    "s3_folder_continued_pretrain_validation": "sec-10k/data/validation",
    "s3_folder_checkpoints_cpt": "sec-10k/checkpoints/cpt",
    "s3_folder_checkpoints_ift": "sec-10k/checkpoints/ift",
    "uc_schema": "<catalog>.<schema-for-workshop>",
    "uc_table": "sec_10k",
    "mlflow_experiment_name_cpt": "/Users/<your-user-id>/<path-to-experiment>",
    "mlflow_experiment_name_ift": "/Users/<your-user-id>/<path-to-experiment>",
    "mcli_secret_scope": "<your-scope>",
    "mcli_api_key": "mcli_api_key",
    "aws_secret_scope": "<your-scope>",
    "s3_access_key": "aws_access_key",
    "s3_secret_access_key": "aws_secret_access_key",
    "cpt_duration": "1ep",
    "ift_duration": "10ba",
    "model_endpoint_name": "<your-endpoint-name>"
}
