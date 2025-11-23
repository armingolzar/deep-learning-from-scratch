import tensorflow as tf

class MyDense(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units


    def build(self, input_shape):
        
        in_features = int(input_shape[-1])
        limit = tf.sqrt(6/ (in_features, self.units))

        self.w = self.add_weight(shape=(in_features, self.units), initializer=tf.random_uniform_initializer(-limit, limit), trainable=True)

        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b
        

