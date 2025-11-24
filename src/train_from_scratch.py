import tensorflow as tf

class MyDense(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units


    def build(self, input_shape):
        
        in_features = int(input_shape[-1])
        limit = tf.sqrt(6.0 / (in_features, self.units))

        self.w = self.add_weight(shape=(in_features, self.units), initializer=tf.random_uniform_initializer(-limit, limit), trainable=True)

        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b
        

class MyFlatten(tf.keras.layers.Layer):

    def call(self, inputs):
        batch = tf.shape(inputs[0])
        return tf.reshape(inputs, (batch, -1))
    
class MyConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=(1,1), padding="same"):
        super().__init__()
        self.filters = filters
        self.kernel_size = (kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size))
        self.strides = (strides if isinstance(strides, tuple) else (strides, strides))
        self.padding = padding.upper()


    def build(self, input_shape):
        
        kh, kw = self.kernel_size
        in_channels = int(input_shape[-1])

        limit = tf.sqrt(6.0 / (kh * kw * in_channels + self.filters))

        self.kernel = self.add_weight(shape=(kh, kw, in_channels, self.filters), initializer=tf.random_uniform_initializer(-limit, limit), trainable=True)
        self.bias = self.add_weight(shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, inputs):

        kh, kw = self.kernel_size
        sh, sw = self.strides

        