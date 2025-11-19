import tensorflow as tf 
# import cv2

def prepare_dataset_on_gpu(batch_size=64, val_size=5000):
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

    data_val = data_train[-val_size:]
    label_val = label_train[-val_size:]

    data_train = data_train[:-val_size]
    label_train = label_train[:-val_size]


    train_ds = tf.data.Dataset.from_tensor_slices((data_train, label_train)).shuffle(10000).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((data_val, label_val)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((data_test, label_test)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # print("train_ds samples:", sum(1 for _ in train_ds.unbatch()))
    # print("val_ds samples:", sum(1 for _ in val_ds.unbatch()))
    # print("test_ds samples:", sum(1 for _ in test_ds.unbatch()))

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = prepare_dataset_on_gpu()

# for batch_img, batch_label in train_ds.take(1):
#     print(batch_img.shape, batch_label.shape)




