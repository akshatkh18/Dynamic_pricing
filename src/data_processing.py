import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

def load_data():
    df=pd.read_csv(r"data\processed\dynamic_pricing_model_ready.csv")
    return df 
