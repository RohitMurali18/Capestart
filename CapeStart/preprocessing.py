import pandas as pd
df =pd.read_csv(r"C:\Users\Rohit\Downloads\articles.csv")
df.fillna("", inplace=True)
df.isna().sum()