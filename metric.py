from keras import backend as K
import tensorflow as tf
import sys

def top_categorical_accuracy(y_true, y_pred, num_classes):
    print(y_true)
    totalones = K.cast(K.sum(y_true), "int32")
    shape = tf.stack((tf.shape(y_true)[0], 1))
    y_res = tf.fill(shape, 0.0)
    for i in range(num_classes):
        y_res += (tf.gather(y_true,[i], axis=-1) * K.cast(K.in_top_k(y_pred, K.flatten(tf.fill(shape, i)), totalones), K.floatx()) / K.cast(totalones, K.floatx()))
    return y_res





