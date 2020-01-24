from keras import backend as K
import tensorflow as tf

def top_categorical_accuracy(y_true, y_pred, num_classes):
    totalones = K.cast(K.sum(y_true, axis=-1), "int64")
    shape = tf.stack((tf.shape(y_true)[0], 1))
    y_res = tf.fill(shape, 0.0)
    sort = tf.sort(y_pred, axis=-1, direction='DESCENDING')
    for i in range(num_classes):
        y_res += (tf.gather(y_true,i, axis=-1) * K.cast(in_top(y_pred, sort, i, totalones), K.floatx()) / K.cast(totalones, K.floatx()))
    return y_res

def in_top(y_pred, sort, i, k):
    return K.less_equal(tf.gather(sort,k - 1, axis=-1), tf.gather(y_pred, i, axis=-1))
    