import pandas as pd 

file_name = 'logs_summed.csv'

df = pd.DataFrame()
df = pd.read_csv(file_name)

df.drop(index=df.index[:1000], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv(file_name)
