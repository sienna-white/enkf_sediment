
import pandas as pd 



df = pd.read_csv('lisst_data.csv')
df1 = pd.read_csv('lisst_variance.csv')

max_zeros= 10 
zero_counts = (df == 0).sum(axis=1)
cleaned = zero_counts <= 10
filtered1 = df[cleaned]
filtered2 = df1[cleaned]


filtered1.to_csv('lisst_data_cleaned.csv', index=False)
filtered2.to_csv('lisst_variance_cleaned.csv', index=False)