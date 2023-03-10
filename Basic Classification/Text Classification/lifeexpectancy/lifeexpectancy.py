import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

dataset = pd.read_csv(r'VSCode\Non-Prod Projects\lifeexpectancy\life_expectancy.csv')

#print(dataset.head())
#print(dataset.describe())

df = dataset.copy()
df.drop('Country', axis=1)

features = df.iloc[:, :-1]
labels = df.iloc[:,-1]
#print(features)

features = pd.get_dummies(features)

num_features = features.select_dtypes(include=['float64', 'int64'])
num_cols = num_features.columns

print(features.sum())
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=27, test_size=0.3)

ct = ColumnTransformer([('only_numeric', StandardScaler(), num_cols)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)

features_test_scaled = ct.transform(features_test)

my_model = Sequential()
input = InputLayer(input_shape = (features.shape[1], ))
my_model.add(input)
my_model.add(Dense(64, activation='relu'))
my_model.add(Dense(1))

print(my_model.summary())

opt = Adam(learning_rate= 0.01)

my_model.compile(loss='mse', metrics='mae', optimizer = opt)

my_model.fit(features_train_scaled, labels_train, epochs= 40, batch_size= 1, verbose=1)

res_mse, res_mae = my_model.evaluate(features_train_scaled, labels_test, verbose=0)

print(res_mse, res_mae)