from keras import backend as K
import tensorflow as tf
import sys

def top_categorical_accuracy(y_true, y_pred, num_classes):
    totalones = K.cast(K.sum(y_true, axis=-1), "int64")
    shape = tf.stack((tf.shape(y_true)[0], 1))
    y_res = tf.fill(shape, 0.0)
    sort = tf.argsort(y_pred, axis=-1, direction='DESCENDING')
    for i in range(num_classes):
        y_res += (tf.gather(y_true,[i], axis=-1) * K.cast(in_top(y_pred, sort, K.flatten(tf.fill(shape, i)), totalones), K.floatx()) / K.cast(totalones, K.floatx()))
    return y_res

def in_top(y_pred, sort, i, k):
    temp = tf.gather(y_pred, tf.gather(sort,[k], axis=-1), axis=-1)
    return K.less_equal(temp, tf.gather(y_pred, [i], axis=-1))





