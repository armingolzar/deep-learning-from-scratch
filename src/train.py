from src.data_loader import prepare_dataset_on_gpu
from src.model import build_model
from src.utils import training_curve
import src.config as config
import tensorflow as tf 

train_ds, val_ds, test_ds = prepare_dataset_on_gpu()
model = build_model()
model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="Adam",
            metrics=["accuracy"])

history = model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds)

model.save("..\\models\\Cifar10.h5")

training_curve(history)