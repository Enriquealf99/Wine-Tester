
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../data/data.csv')

if df.isnull().sum().any():
    df = df.dropna()

df = df.drop_duplicates()

z_scores = np.abs(zscore(df))
df = df[(z_scores < 3).all(axis=1)]

X = df.drop('quality', axis=1)
y = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

df.to_csv('../data/preprocessed_winequality-red.csv', index=False)
