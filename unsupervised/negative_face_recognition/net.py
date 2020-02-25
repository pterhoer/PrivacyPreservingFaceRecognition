# Author: Marco Huber, 2019

from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adadelta

def net(n_ID):
    
    """
        simple sample net to be used to train larger representations of given representations
        while retaining the identity information
        
        return: keras model
    """
    
    model = Sequential()
    model.add(Dense(256, input_shape=(128,), activation="relu", kernel_initializer='random_uniform', name='input'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Dense(512, activation="relu", kernel_initializer='random_uniform', name='second'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Dense(4096, activation="tanh", kernel_initializer='random_uniform', name='emb'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Dense(n_ID, activation='softmax', name='softmax'))
    
    # compile
    optimizer = Adadelta(lr=0.1)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    return model

def get_representation_model(model):
    """
    generates/predicts expanded embeddings of the given input features/templates
    """
    
    representation_model = Model(inputs= model.input, outputs=model.get_layer("emb").output)
    return representation_model