import pandas as pd 

file_name = 'logs_summed.csv'

df = pd.DataFrame()
df = pd.read_csv(file_name)

df1 = df.drop(index=df.index[:1000], axis=0)
df1 = df1.drop(index=df.index[2000:], axis=0)
df1 = df1.reset_index(drop=True)
df1.to_csv('logs_summed_1.csv')

df2 = df.drop(index=df.index[:2000], axis=0)
df2 = df2.reset_index(drop=True)
df2.to_csv('logs_summed_2.csv')

df1 = df.drop(index=df.index[1000:], axis=0)
df1 = df1.reset_index(drop=True)
df1.to_csv('logs_summed_adv.csv')