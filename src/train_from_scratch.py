import tensorflow as tf
from data_loader import prepare_dataset_on_gpu
from utils import training_curve_ctl

class MyDense(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        
        in_features = int(input_shape[-1])
        limit = tf.sqrt(6.0 / (in_features + self.units))

        self.w = self.add_weight(name="dense_weights", shape=(in_features, self.units), initializer=tf.random_uniform_initializer(-limit, limit), trainable=True)

        self.b = self.add_weight(name="dense_bias", shape=(self.units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
        

class MyFlatten(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        return tf.reshape(inputs, (batch, -1))
    
    def get_config(self):
        config = super().get_config()
        return config


class MyConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=(1,1), padding="same", name="MyConv2D"):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size))
        self.strides = (strides if isinstance(strides, tuple) else (strides, strides))
        self.padding = padding.upper()


    def build(self, input_shape):
        
        kh, kw = self.kernel_size
        in_channels = int(input_shape[-1])

        limit = tf.sqrt(6.0 / (kh * kw * in_channels + self.filters))

        self.kernel = self.add_weight(name="kernel_weights", shape=(kh, kw, in_channels, self.filters), initializer=tf.random_uniform_initializer(-limit, limit), trainable=True)
        self.bias = self.add_weight(name="kernel_bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):

        kh, kw = self.kernel_size
        sh, sw = self.strides

        patches = tf.image.extract_patches(images=inputs, sizes=[1, kh, kw, 1], strides=[1, sh, sw, 1], rates=[1, 1, 1, 1], padding=self.padding)
        # patches shape: (batch, out_h, out_w, kH*kW*in_channels)

        batch = tf.shape(patches)[0]
        out_h = tf.shape(patches)[1]
        out_w = tf.shape(patches)[2]
        patch_dim = tf.shape(patches)[3]

        # Reshape patches so matmul works
        patches_flat = tf.reshape(patches, (batch * out_h * out_w, patch_dim))

        # Flatten kernel
        flat_kernel = tf.reshape(self.kernel, (patch_dim, self.filters))

        # Dense-style multiply → conv output
        conv = tf.matmul(patches_flat, flat_kernel)
        conv = tf.reshape(conv, (batch, out_h, out_w, self.filters))

        return conv + self.bias
    
    def get_config(self):
        config = super().get_config()
        config.update({"filters" : self.filters, "kernel_size" : self.kernel_size, "strides" : self.strides, "padding" : self.padding.lower()})
        return config
    
class MyModel(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        self.conv1a = MyConv2D(32, (3, 3))
        self.conv1b = MyConv2D(32, (3, 3))
        self.conv2a = MyConv2D(64, (3, 3))
        self.conv2b = MyConv2D(64, (3, 3))
        self.flatten = MyFlatten()
        self.dense1 = MyDense(64)
        self.dense2 = MyDense(10)
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs, training=False):

        conv1 = self.conv1a(inputs)
        act1 = self.relu(conv1)
        conv2 = self.conv1b(act1)
        act2 = self.relu(conv2)
        maxpool1 = self.maxpool(act2)

        conv3 = self.conv2a(maxpool1)
        act3 = self.relu(conv3)
        conv4 = self.conv2b(act3)
        act4 = self.relu(conv4)

        flatten = self.flatten(act4)
        dense1 = self.dense1(flatten)
        act5 = self.relu(dense1)
        output = self.dense2(act5)

        return output
    
    def get_config(self):
        config = super().get_config()
        return config


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
val_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_loss_metric = tf.keras.metrics.Mean()

best_val_acc = 0.0
EPOCHS = 20

history = {"loss" : [], "accuracy" : [], "val_loss" : [], "val_accuracy" : []}

train_ds, val_ds, test_ds = prepare_dataset_on_gpu()

model = MyModel()

@tf.function
def train_step(data, label):
    with tf.GradientTape() as tape:
        model_output = model(data, training=True)
        loss = loss_fn(label, model_output)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc.update_state(label, model_output)

    return loss

@tf.function
def val_step(val_data, val_label):
    val_output = model(val_data, training=False)
    v_loss = val_loss_fn(val_label, val_output)
    val_acc.update_state(val_label, val_output)
    val_loss_metric.update_state(v_loss)


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

    print(f"- loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        model.save("..\\models\\best_Cifar_from_scratch", save_format="tf")
        print(f"Saved new best model ---> {best_val_acc}")

    history["loss"].append(train_loss)
    history["accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

training_curve_ctl(history)

# This function is for printing some useful information about a model
def model_info(model):

    # 1. This is in layer stage and it shows top layer variables
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.__class__.__name__}")
        print(f" - Unique TF name: {layer.name}")

        trainable_params = sum(tf.size(v).numpy() for v in layer.trainable_variables)
        non_trainable_params = sum(tf.size(v).numpy() for v in layer.non_trainable_variables)
        total_params = trainable_params + non_trainable_params

        print(f" - Trainable params: {trainable_params}")
        print(f" - Non-trainable params: {non_trainable_params}")
        print(f" - Total params: {total_params}")

        print(" - Variables:")
        for v in layer.variables:
            print(f" • {v.name} | shape = {v.shape} | trainable: {v.trainable}")

        print("------------------------------------------------------------")

    # 2. This is in model stage and it shows all layers and variables including nested ones
    print("=== All model variables ===")
    for v in model.variables:
        print(f"{v.name} | shape: {v.shape} | trainable: {v.trainable}")

model_info(model)

### Attention when you want to save your model with custom layers, 
# you can pass a name for the layer but you have to pass a name for each weight that you create in build method of that layer
# also you need to define get_config() method and add all new arguments and prameters of __init__ to super().get_config()
# so tensorflow can serialize and gather every customize thing in the layer and save it for you to load and use
# look at above code which customize layers has name or not for theirselves but all of them have a name for their weights and have get_config() method