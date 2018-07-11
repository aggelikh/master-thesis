import pandas as pd
import operator
import sys
import numpy


#eisagw to arxeio, diwxnw timestamps and edges kai ta dipla.
df_init = pd.read_csv("twitterex_ut.txt", sep=" ", header=None, names=["Tags", "Users", "Edges", "Timestamps"])

df = df_init.drop(['Edges', 'Timestamps'], axis=1)

df = df.drop_duplicates(keep="first")

df['count'] = df.groupby('Users')['Users'].transform(pd.Series.value_counts)

sample = df.sort_values('count', ascending=False)


sample['count']
print(df.shape)

