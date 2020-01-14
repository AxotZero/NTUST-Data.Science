import pandas as pd
import numpy as np
import sklearn.preprocessing as sk_preprocessing

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 100)

df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

# Drop Columns
dropColumns = ['Attribute2', 'Attribute8', 'Attribute10', 'Attribute11', 'Attribute12', 'Attribute14', 'Attribute16', 'Attribute18', 'Attribute20']
df = df.drop(columns=dropColumns)
df2 = df2.drop(columns=dropColumns)

# Drop row with nan values in it.
df = df.dropna()

# Get Month
df['Attribute1'] = pd.DatetimeIndex(df['Attribute1']).month
df2['Attribute1'] = pd.DatetimeIndex(df2['Attribute1']).month

# Replace Yes/NO to 1/0
df = df.replace(to_replace='Yes', value = 1)
df = df.replace(to_replace='No', value = 0)
df2 = df2.replace(to_replace='Yes', value = 1)
df2 = df2.replace(to_replace='No', value = 0)

# Save all
df.to_csv('preprocessed_train_All.csv')
df2.to_csv('preprocessed_test_A.csv')

# Balance
Yes = df[df['Attribute23'] == 1]
No = df[df['Attribute23'] == 0].sample(n=len(Yes), random_state=1)

df = pd.concat([Yes, No]).sample(frac=1, random_state=20)

# Recoding Label
label = pd.DataFrame()
label['L'] = df['Attribute23']
X = label.values
label = pd.DataFrame(label.values)
df = df.drop(columns=['Attribute23'])

# Normalize
# X = sk_preprocessing.normalize(df.values, norm='max', axis=0)
# df = pd.DataFrame(X)

# Save
df.to_csv('preprocessed_train_A.csv')
label.to_csv('preprocessed_train_A_label.csv')