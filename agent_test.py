import os
import pandas as pd

def df_test():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_key.json'
    path = "gs://kaggle-exercise-csv/kaggle-exercise"
    df = pd.read_csv(path)
    return df.shape[0]