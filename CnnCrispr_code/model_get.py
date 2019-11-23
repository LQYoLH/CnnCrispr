from __future__ import print_function
from keras.layers import Bidirectional, LSTM, BatchNormalization
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D



def modelSelection(model, MODEL_select):
    model_tmpl = ["Class", "model_Conv_LSTM","Class_noLSTM","Class_noConv", "Class_noBatchNor",
                  "Class_noDropout", "Reg"]
    if MODEL_select == model_tmpl[0]:
        return model1(model)
    if MODEL_select == model_tmpl[1]:
        return model_Conv_LSTM(model)
    if MODEL_select == model_tmpl[2]:
        return model1_noLSTM(model)
    if MODEL_select == model_tmpl[3]:
        return model1_noConv(model)
    if MODEL_select == model_tmpl[4]:
        return model1_noBatchNor(model)
    if MODEL_select == model_tmpl[5]:
        return model1_noDropout(model)
    if MODEL_select == model_tmpl[6]:
        return model1_reg(model)


def model1(model_ini):
    # The benchmark model CnnCrispr for Classification schema
    print("model1 loaded with 1 biLSTM, 5 conv and 2 dense")
    model_message = "Dropout 0.3,biLSTM.40, Conv1D.[10,20,40,80,100],  Dense[20,2], BatchNormalization,Activition='relu'"
    model = model_ini
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(Activation('relu'))

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model, model_message


def model_Conv_LSTM(model_ini):
    # Comparison model by change the order of convolution layer and recurrent layer
    print("model1 loaded with 5 conv, 1 biLSTM and 2 dense")
    model_message = "Dropout 0.3,biLSTM.40, Conv1D.[10,20,40,80,100],  Dense[20,2], BatchNormalization"
    model = model_ini

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model, model_message

def model1_noLSTM(model_ini):
    # Comparison model by taken out the LSTM layer
    print("model1 loaded with 5 conv and 2 dense")
    model_message = "Dropout 0.2, Conv1D.[10,20,40,80,100],  Dense[20,2], BatchNormalization"
    model = model_ini

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model, model_message


def model1_noConv(model_ini):
    # Comparison model by taken out the convolution layer
    print("model1 loaded with 1 biLSTM and  dense")
    model_message = "Dropout 0.3,biLSTM.40,  Dense[1000,100,20,2], BatchNormalization"
    model = model_ini
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model, model_message


def model1_noBatchNor(model_ini):
    # Comparison model by taken out the BatchNormalization layer
    print("model1 loaded with 5 conv, 1 biLSTM and 2 dense")
    model_message = "Dropout 0.3,biLSTM.40, Conv1D.[10,20,40,80,100],  Dense[20,2]"
    model = model_ini
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(Activation('relu'))

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model, model_message


def model1_noDropout(model_ini):
    # Comparison model by taken out the Dropout layer
    print("model1 loaded with 5 conv, 1 biLSTM and 2 dense")
    model_message = "biLSTM.40, Conv1D.[10,20,40,80,100],  Dense[20,2], BatchNormalization"
    model = model_ini
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(Activation('relu'))

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model, model_message


def model1_reg(model_ini):
    # he benchmark model CnnCrispr for Regression schema
    print("model1 loaded with 1 biLSTM, 5 conv and 2 dense")
    model_message = "Dropout 0.3,biLSTM.40, Conv1D.[10,20,40,80,100],  Dense[20,1], BatchNormalization"
    model = model_ini
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(40, return_sequences=True)))
    model.add(Activation('relu'))

    model.add(Conv1D(10, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(20, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(40, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(80, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv1D(100, (5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model, model_message
