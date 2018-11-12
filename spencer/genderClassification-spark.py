# use findspark to find all the newly installed modules
import findspark
findspark.init()

# Numerical Imporst 
import pandas as pd
import numpy as np
import scipy 

# Plotting 
import matplotlib.pyplot as plt
#import seaborn as sns

# Python 
import os

# mllib
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# scipy
from scipy.cluster import hierarchy as hc # for dendograms


#build datasets
df_train_pro = pd.read_table(f'{os.getcwd()}/train_pro.tsv', 
                           delim_whitespace=True,
                           low_memory=False,).T
df_test_pro = pd.read_table(f'{os.getcwd()}/test_pro.tsv', 
                           delim_whitespace=True,
                           low_memory=False,).T
df_train_cli = pd.read_csv(f'{os.getcwd()}/train_cli.tsv', 
                           delim_whitespace=True,
                           low_memory=False,)
df_test_cli = pd.read_csv(f'{os.getcwd()}/test_cli.tsv', 
                           delim_whitespace=True,
                           low_memory=False,)
df_train_mislabel = pd.read_csv(f'{os.getcwd()}/sum_tab_1.csv', 
                           low_memory=False,)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

# Come back to the way you handle this NA, sophisticated way will imporve by at least 5% 
train_pro = df_train_pro.copy(deep=True)
train_pro = train_pro.fillna(train_pro.mean())
train_pro.index.name = 'sample'


test_pro = df_test_pro.copy(deep=True)
test_pro = test_pro.fillna(test_pro.mean())
test_pro.index.name = 'sample'

train_cli = df_train_cli.copy(deep=True)
train_cli = train_cli.set_index('sample')
train_cli = train_cli.replace({'gender': {'Male':0, 'Female':1},
                              'msi': {'MSI-Low/MSS':0, 'MSI-High':1}})

test_cli = df_test_cli.copy(deep=True)
test_cli = test_cli.set_index('sample')
test_cli = test_cli.replace({'gender': {'Male':0, 'Female':1},
                              'msi': {'MSI-Low/MSS':0, 'MSI-High':1}})

train_mislabel = df_train_mislabel.copy(deep=True)
train_mislabel = train_mislabel.set_index('sample')

train_pro.reset_index(drop=True, inplace=True)
train_cli.reset_index(drop=True, inplace=True)
train_mislabel.reset_index(drop=True, inplace=True)

train_combined = pd.concat([train_mislabel, train_cli, train_pro], axis=1)

train_combined_correct = train_combined.loc[train_combined['mismatch'] == 0]
X_correct = train_combined_correct.drop(['mismatch'], axis=1, inplace=False)
X_correct.reset_index(drop=True, inplace=True)

gender_correct = X_correct['gender']
msi_correct = X_correct['msi']
X_correct = X_correct.drop(['gender', 'msi'], axis=1, inplace=False)

columns = X_correct.columns

# split data
#X_gender_train, X_gender_valid, y_gender_train, y_gender_valid = train_test_split(X_correct.values.astype(int), gender_correct, test_size=0.3)

#train, test = data.randomSplit([0.9, 0.1], seed=12345)
