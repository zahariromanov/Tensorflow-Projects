import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

hf = pd.read_csv('Tensorflow-Projects\Basic Classification\heart_failure\heart_failure.csv')
print(hf.info())

print('Classes and number of values in the dataset' ,Counter(hf['death_event']))

y = hf['death_event']
x = hf[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

x = pd.get_dummies(x)

# uses sklearn.model_selection.train_test_split() method to split data into training figures
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# uses ColumnTransformer and Normalizer
ct = ColumnTransformer([('numeric', Normalizer(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

# trains instance of ct on X_train
X_train = ct.fit_transform(X_train)

# scales test data using trained scaler
X_test = ct.transform(X_test)

# init labelencoder and fit instance to Y_train and Y_test
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.transform(Y_test.astype(str))

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# build model
model = Sequential()
## create input layer
model.add(InputLayer(input_shape=(X_train.shape[1],)))
### create hidden layer
model.add(Dense(12, activation='relu'))
#### create output layer
model.add(Dense(2, activation='softmax'))
##### compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
##### fit the model to the X_train and Y_train training data, process with 100 epochs and a batch size of 16
model.fit(X_train, Y_train, epochs = 100, batch_size = 16, verbose=1)
###### evaluate model
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Loss', loss, 'Accuracy:', acc)

# make prediction
y_estimate = model.predict(X_test, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)
print(classification_report(y_true, y_estimate))
