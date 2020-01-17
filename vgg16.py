# vgg 16 (16 layers with weights)
import functools
import keras
import keras.metrics as metrics
import metric
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam

def vggmodel(num_classes, size, compiled = True):
    model = Sequential()
    model.add(Conv2D(input_shape=(size[1],size[0],3),filters=64,kernel_size=(3,3),padding="same", activation="relu")) # needs adjustment input size
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name="last_conv"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    if(compiled):
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_classes, activation="sigmoid")) # needs other kind of layer for chance per label

        top3_acc = functools.partial(metric.top_categorical_accuracy, num_classes=num_classes)
        top3_acc.__name__ = 'top3_accuracy'
        opt = Adam(lr=0.001)
        model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=[top3_acc, metrics.categorical_accuracy])
    return model



