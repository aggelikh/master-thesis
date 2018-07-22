import pandas as pd
import sklearn
from sklearn.utils import shuffle


# eisagw to arxeio, diwxnw timestamps and edges kai ta dipla.
df_init = pd.read_csv("twitterex_ut.txt", sep=" ", header=None, names=["Tags", "Users", "Edges", "Timestamps"])

df = df_init.drop(['Edges', 'Timestamps'], axis=1)

# drop duplicate values
df = df.drop_duplicates(keep="first")

# make a new column which counts the frequency of users. It is grouped by users
df['count'] = df.groupby('Users')['Users'].transform(pd.Series.value_counts)

# sort this dataframe by count so to have at the top the most tagged users
sample = df.sort_values('count', ascending=False)

# split the sample, keep only columns=[Users, count] and drop double multiple records so to have each user and its frequency
user_count = sample.loc[:, ('Users', 'count')].drop_duplicates(keep="first")

# keep the first 50000 users and join two dataframes, user_count and sample, by key=count
user_count = user_count[:50000].join(sample.set_index('Users'), on='Users', lsuffix="_user", rsuffix="_df")

# drop column=count and my semifinal dataframe consists of 50000 most tagged users with their tags
semifinal = user_count.drop(['count_user', 'count_df'], axis=1)

print(len(df["Users"].unique().tolist()))
print(len(df["Tags"].unique().tolist()))
print()

print(len(semifinal["Users"].unique().tolist()))
print(len(semifinal["Tags"].unique().tolist()))
print()

# the same procedure in order as above in order to keep the first 10000most used tags. My initial dataframe is "semifinal" and result to final!!!!
semifinal['count'] = semifinal.groupby('Tags')['Tags'].transform(pd.Series.value_counts)
tag_count = semifinal.sort_values('count', ascending=False)

tag_count = tag_count.loc[:, ('Tags', 'count')].drop_duplicates(keep="first")

tag_count = tag_count[:10000].join(semifinal.set_index('Tags'), on='Tags', lsuffix="_tag", rsuffix="_df")

final = tag_count.drop(['count_tag', 'count_df'], axis=1)

# shuffle my final dataframe so to not be sorted
final = sklearn.utils.shuffle(final)



print(len(final['Users'].unique().tolist()))
print(len(final['Tags'].unique().tolist()))
print(final.info())

final.to_csv('final.csv', sep=' ')
