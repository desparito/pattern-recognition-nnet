import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

def fcnmodel(num_classes):
    # get input objects
    # get genres as input labels
    model = sequential()
    netSize = 120
    model.Add(Dense(netSize, activation = 'relu', input_shape = (80, 1)))
    model.Add(Dense(netSize, activation = 'relu'))
    model.Add(Dense(netSize, activation = 'relu'))
    model.add(Dense(units=num_classes, activation="sigmoid"))

    # TODO: construct combined neural net in other script somewhere, combine the total
    # TODO: create input stream for combined neural net

    top3_acc = functools.partial(metric.top_categorical_accuracy, num_classes=num_classes)
    top3_acc.__name__ = 'top3_accuracy'
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[top3_acc])
    return model
    

    