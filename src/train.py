from data_loader import prepare_dataset_on_gpu
from model import build_model
from utils import training_curve
import tensorflow as tf 

train_ds, val_ds, test_ds = prepare_dataset_on_gpu()
model = build_model()
model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="Adam",
            metrics=["accuracy"])

history = model.fit(train_ds, batch_size=64, epochs=20, validation_data=val_ds)

model.save("..\\models\\Cifar10.h5")

training_curve(history)