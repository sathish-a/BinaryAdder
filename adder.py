import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense

x = 0
y = 0
x_train = 0
y_train = 0
x_test = 0
y_test = 0

model = None


def init():
    global model
    model = Sequential()
    model.add(Dense(units=128, input_dim=8, activation='sigmoid'))
    model.add(Dense(units=64, activation='sigmoid'))
    model.add(Dense(units=8, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


def generate(v):
    x_ = []
    for i in range(254):
        y_ = []
        z = bin(i + v)[2:].zfill(8)
        for j in z:
            y_.append(int(j))
        x_.append(y_)
    return np.array(x_)


def get_dataset():
    return shuffle(generate(0), generate(1), random_state=0)


# to train
def learn():
    global x, y, x_train, y_train, x_test, y_test
    x, y = get_dataset()
    x_train = x[0:231]
    y_train = y[0:231]
    x_test = x[231:255]
    y_test = y[231:255]
    model.fit(x_train, y_train, epochs=500, batch_size=5)


def save():
    model.save_weights("modeladd.h5")

# loss: 3.7280e-04 - acc: 1.0000


def load():
    model.load_weights("modeladd.h5")


def next(val):
    if val <= 254:
        out_l = []
        inp_l = []
        b = bin(val)[2:].zfill(8)
        for k in b:
            inp_l.append(int(k))
        out_l.append(inp_l)
        out_l = np.array(out_l)
        classes = model.predict(out_l)
        output = []
        for u in classes:
            li = []
            for z in u:
                if z >= 0.1:
                    li.append(1)
                else:
                    li.append(0)
            output.append(li)
        output = np.array(output)
        print("Input:", out_l, "output:", output)
    else:
        print("range exceeds")


init()
load()
next(231)
