import tensorflow as tf 
# import cv2
(data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()


# print(label_train[0])
# cv2.imshow("image", data_train[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(data_train[0].shape)
# print(data_test.shape)

def preprocess(image, label):
    image = tf.cast(image, tf.float32)/255.0
    return image, label

val_size = 5000

data_train = data_train[:-val_size]
label_train = label_train[:-val_size]

data_val = data_train[-val_size:]
label_val = label_train[-val_size:]

train_ds = tf.data.Dataset.from_tensor_slices((data_train, label_train))

