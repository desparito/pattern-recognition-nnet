import functools
import keras
import metric
import keras.models as models
import keras.metrics as metrics
from keras.layers import Dense, Flatten, BatchNormalization, Concatenate
from keras.models import Sequential
from keras.optimizers import Adam


def fcnmodel(num_classes, yolo_size, img_model):
    # get input objects
    # get genres as input labels
    yolo_model = Sequential()
    yolo_model.add(Dense(units=yolo_size,activation="relu", input_shape = (1, yolo_size)))
    yolo_model.add(Dense(units=1024,activation="relu"))
    yolo_model.add(Dense(units=4096,activation="relu"))
    yolo_model.add(Flatten())
    yolo_model.add(BatchNormalization())

    img_model.add(Flatten())
    img_model.add(BatchNormalization())

    model = Concatenate()([yolo_model.output, img_model.output])
    #model = Flatten()(model)
    model = Dense(units=4096, activation = 'relu')(model)
    model = Dense(units=num_classes, activation="sigmoid")(model)

    model = models.Model([yolo_model.input, img_model.input], model, name='fcnet')

    top3_acc = functools.partial(metric.top_categorical_accuracy, num_classes=num_classes)
    top3_acc.__name__ = 'top3_accuracy'
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[top3_acc, metrics.categorical_accuracy])
    return model
    

    