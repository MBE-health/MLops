import os
import pandas as pd
import json

def df_test():
    CREDENTIALS = {
    "type": "service_account",
    "project_id": os.getenv('GCP_project_id'),
    "private_key_id": os.getenv('GCP_private_key_id'),
    "private_key": os.getenv('GCP_private_key'),
    "client_email": os.getenv('GCP_client_email'),
    "client_id": os.getenv('GCP_client_id'),
    "auth_uri": os.getenv('GCP_auth_uri'),
    "token_uri": os.getenv('GCP_token_uri'),
    "auth_provider_x509_cert_url":os.getenv('GCP_auth_provider_x509_cert_url'),
    "client_x509_cert_url":os.getenv('GCP_client_x509_cert_url'),
    }
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =  json.dumps(CREDENTIALS)
    path = "gs://kaggle-exercise-csv/kaggle-exercise"
    df = pd.read_csv(path)
    return df.shape[0]