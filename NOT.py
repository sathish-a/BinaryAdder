from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

x_train = [[0, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0],
           [0, 1, 0, 1, 1], [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 0, 0, 0, 0],
           [1, 0, 0, 0, 1], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0],
           [1, 0, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 1, 0, 1, 1], [1, 1, 1, 0, 0],
           [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
y_train = [[1, 1, 0, 1, 0], [1, 1, 0, 0, 1], [1, 1, 0, 0, 0], [1, 0, 1, 1, 1], [1, 0, 1, 1, 0], [1, 0, 1, 0, 1],
           [1, 0, 1, 0, 0], [1, 0, 0, 1, 1], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 1, 1, 1],
           [0, 1, 1, 1, 0], [0, 1, 1, 0, 1], [0, 1, 1, 0, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1],
           [0, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1],
           [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]

x_test = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 1, 0, 0]]

y_test = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 1], [1, 1, 1, 0, 0], [1, 1, 0, 1, 1]]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)


x_train = x_train.reshape(len(x_train), 5)
y_train = y_train.reshape(len(y_train), 1)
x_test = x_test.reshape(len(x_test), 1)
y_test = y_test.reshape(len(y_test), 1)


model = Sequential()
model.add(Dense(units=128, input_dim=5, activation='sigmoid'))
model.add(Dense(units=64, activation='sigmoid'))
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dense(units=16, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(units=5, activation="sigmoid"))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=750, batch_size=5)
model.save_weights("model.h5")

model.load_weights("model.h5")
classes = model.predict(x_test, batch_size=10)

output = []
for x in classes:
    li = []
    for y in x:
        if y >= 0.1:
            li.append(1)
        else:
            li.append(0)
    output.append(li)
print(output)
