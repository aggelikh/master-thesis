import numpy as np
import pandas as pd

df = pd.read_csv("final.csv", sep=" ", header=None, names=["Tags", "Users"])

print(df.info())

print(len(df["Users"].unique().tolist()))
print(len(df["Tags"].unique().tolist()))

print(df.iloc[:100])