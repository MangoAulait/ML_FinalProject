"""
Author: Luyao Chen
"""


import pandas as pd

def clean_data_process(df):
    df["screen_name"]=df["screen_name"].apply(removeQuotation)

    return df

def removeQuotation(name):
    if name.startswith('"') and name.endswith('"'):
        thename = name[1:-1]
        return thename
    return name
