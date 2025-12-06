from src.data_loader import prepare_dataset_on_gpu
from src.model import build_model
from src.utils import training_curve
import src.config as config
import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, ModelCheckpoint


train_ds, val_ds, test_ds = prepare_dataset_on_gpu()
model = build_model()
model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="Adam",
            metrics=["accuracy"])

# you can pass the min_delta argument for EarlyStopping and ReduceLROnPlateau and define what is improvement, if you dont pass, it just need a better result even it is extremely small.
callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1), 
             ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_delta=1e-4, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath=".\\models\\best_model_callback.h5", monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1),
             CSVLogger(filename=".\\models\\log\\model_callback.csv", append=False, separator=",")]

history = model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds, callbacks=callbacks)

model.save("..\\models\\Cifar10.h5")

training_curve(history)