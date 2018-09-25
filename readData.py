# This file reads the initial dataset from csv, drop duplicates records,
# keep only "Users" and "Tags" columns. Then, it keeps the 5000 most used users. After this, from these users
# it keeps the 1000 most used tags and concluded to a dataframe consisting of about 86000 lines
# and there are approximately 4500 unique users and 1500 unique tags.

import pandas as pd
import sklearn.utils

# read the file, delete timestamps, edges and duplicates
df_init = pd.read_csv("twitterex_ut.txt", sep=" ", header=None, names=["Tags", "Users", "Edges", "Timestamps"])

df = df_init.drop(['Edges', 'Timestamps'], axis=1)
# print(df.info())
# drop duplicate values
df = df.drop_duplicates(keep="first")

# make a new column which counts the frequency of users. It is grouped by users
df['count'] = df.groupby('Users')['Users'].transform(pd.Series.value_counts)

# sort this dataframe by count so to have at the top the most tagged users
sample = df.sort_values('count', ascending=False)

# split the sample, keep only columns=[Users, count] and drop double multiple records so to have each user and its
# frequency
user_count = sample.loc[:, ('Users', 'count')].drop_duplicates(keep="first")

# keep the first 1% users and join two dataframes, user_count and sample, by key=count
user_count = user_count[:5304].join(sample.set_index('Users'), on='Users', lsuffix="_user", rsuffix="_df")

# drop column=count and my semifinal data frame consists of 5000
#  most tagged users with their tags
semifinal = user_count.drop(['count_user', 'count_df'], axis=1)

# print(len(df["Users"].unique().tolist()))
# print(len(df["Tags"].unique().tolist()))
# print()

# the same procedure in order as above in order to keep the first 1000 most used tags. My initial data frame is
# "semifinal" and result to final!!!!
semifinal['count'] = semifinal.groupby('Tags')['Tags'].transform(pd.Series.value_counts)
tag_count = semifinal.sort_values('count', ascending=False)

tag_count = tag_count.loc[:, ('Tags', 'count')].drop_duplicates(keep="first")

# keep the first 1% users and join two dataframes, user_count and sample, by key=count
tag_count = tag_count[:1752].join(semifinal.set_index('Tags'), on='Tags', lsuffix="_tag", rsuffix="_df")

final = tag_count.drop(['count_tag', 'count_df'], axis=1)

# shuffle my final data frame so to not be sorted
final = sklearn.utils.shuffle(final)

# print(len(final['Users'].unique().tolist()))
# print(len(final['Tags'].unique().tolist()))
# print(final.info())

final.to_csv('final.csv', sep=' ')
