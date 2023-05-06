import pandas as pd
import os 

def df_test():
    url =  os.getenv('GCP_kaggle_exercise')
    df = pd.read_csv(url)
    return df.shape[0]