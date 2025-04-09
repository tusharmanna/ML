#I am developing ANN for obesity prediction#import all modules related tensor and keras and scikit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense

#load the dataset
df = pd.read_csv('obesity.csv')

df.head()

#check for missing values
df.isnull().sum()

#check for duplicates
df.duplicated().sum()

#check for correlation
df.corr()

#visualize the data
sns.heatmap(df.corr(), annot=True)

#model training
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#build the model
#update code to use softmax

model = Sequential()
model.add(Dense(12, input_dim=12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#generate code for earlystopping
from keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)




model.fit(X_train, y_train, epochs=150, batch_size=10, validation_split=0.2, callbacks=[callback])

#make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

#evaluate the model
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print('Confusion Matrix:', cm)
print('Accuracy:', ac)

#Save the model
model.save('obesity.h5')

#load the model
model = load_model('obesity.h5')
