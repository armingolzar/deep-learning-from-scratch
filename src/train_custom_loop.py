from data_loader import prepare_dataset_on_gpu
from model import build_model
from utils import training_curve_ctl
import tensorflow as tf



loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
val_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam()


train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_loss_metric = tf.keras.metrics.Mean()
best_val_acc = 0.0
EPOCHS = 20

history = {"loss":[], "accuracy":[], "val_loss":[], "val_accuracy":[]}

train_ds, val_ds, test_ds = prepare_dataset_on_gpu()
model = build_model()

@tf.function
def train_step(data, label):

    with tf.GradientTape() as tape:
        output = model(data, training=True)
        loss = loss_fn(label, output)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc.update_state(label, output)

    return loss

@tf.function
def val_step(data_val, label_val):
    output_val = model(data_val, training=False)
    loss_val = val_loss_fn(label_val, output_val)
    val_acc.update_state(label_val, output_val)
    val_loss_metric.update_state(loss_val)


for epoch in range(EPOCHS):
    print(f"\n Epoch {epoch+1}/{EPOCHS}")

    train_acc.reset_state()
    val_acc.reset_state()
    val_loss_metric.reset_state()

    for images, labels in train_ds:
        loss = train_step(images, labels)

    for val_images, val_labels in val_ds:
        val_step(val_images, val_labels)

    train_loss = loss.numpy()
    val_loss = val_loss_metric.result().numpy()
    train_accuracy = train_acc.result().numpy()
    val_accuracy = val_acc.result().numpy()

    print(
        f"- loss: {train_loss:.4f} "
        f"- accuracy: {train_accuracy:.4f} "
        f"- val_loss: {val_loss:.4f} "
        f"- val_accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy >  best_val_acc:
        best_val_acc = val_accuracy
        model.save("..\\models\\best_ctl_model.h5")
        print("Saved new best model")

    history["loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)


training_curve_ctl(history)
