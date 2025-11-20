import tensorflow as tf 
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

input_layer = Input(shape=(32, 32, 3), name="input_layer")
conv1 = Conv2D(32, (3, 3), activation="relu", name="conv1")(input_layer)
conv2 = Conv2D(32, (3, 3), activation="relu", name="conv2")(conv1)
maxpool1 = MaxPooling2D(name="maxpool1")(conv2)
conv3 = Conv2D(64, (3, 3), activation="relu", name="conv3")(maxpool1)
conv4 = Conv2D(64, (3, 3), activation="relu", name="conv4")(conv3)
flatten = Flatten(name="flatten")(conv4)
dense1 = Dense(64, activation="relu", name="dense1")(flatten)
output = Dense(10, activation="softmax", name="output")(dense1)

model = Model(inputs=input_layer, outputs=output)

model.summary()





