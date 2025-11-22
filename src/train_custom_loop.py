from data_loader import prepare_dataset_on_gpu
from model import build_model
import tensorflow as tf


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc = tf.keras.metrics.SparseCategoricalAccuracy()


train_ds, val_ds, test_ds = prepare_dataset_on_gpu()
model = build_model()

@tf.function
def train_step(data, label):

    with tf.GradientTape() as tape:
        output = model(data, training=True)
        loss = loss_fn(label, output)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradient(zip(grads, model.trainable_variables))
    train_acc.update_state(label, output)

    return loss

@tf.function
def val_step(data_val, label_val):
    output_val = model(data_val, training=False)
    val_acc.update_state(label_val, output_val)


EPOCHS = 20

for epoch in range(EPOCHS):
    print(f"\n Epoch {epoch+1}/{EPOCHS}")

    train_acc.reset_state()
    val_acc.reset_state()

    for images, labels in train_ds:
        loss = train_step(images, labels)

    for val_images, val_labels in val_ds:
        val_step(val_images, val_labels)

    print(f"- loss: {loss.numpy():.4f} - acc: {train_acc.result():.4f} - val_acc: {val_acc.result():.4f}")

