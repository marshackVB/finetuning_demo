# Databricks notebook source
config = {
    "max_length": 2048,
    "eval_prompts": [],
    "s3_bucket": "s3://timl-mosaic-sandbox",
    "s3_folder_continued_pretrain_train": "sec-10k/data/train",
    "s3_folder_continued_pretrain_validation": "sec-10k/data/validation",
    "s3_folder_checkpoints_cpt": "sec-10k/checkpoints/cpt",
    "s3_folder_checkpoints_ift": "sec-10k/checkpoints/ift",
    "uc_schema": "main.timl_mosaic",
    "uc_table": "sec_10k",
    "mlflow_experiment_name_cpt": "/Users/tim.lortz@databricks.com/MosaicML/MLflow_experiments/mpt-7b-dolly_hhrlhf",
    "mlflow_experiment_name_ift": "/Users/tim.lortz@databricks.com/MosaicML/MLflow_experiments/mpt-7b-dolly_hhrlhf",
    "mcli_secret_scope": "timl_scope",
    "mcli_api_key": "mcli_api_key",
    "aws_secret_scope": "timl_scope",
    "s3_access_key": "aws_access_key",
    "s3_secret_access_key": "aws_secret_access_key",
    "cpt_duration": "1ep",
    "ift_duration": "10ba",
    "model_endpoint_name": "mlc_test_model"
}
