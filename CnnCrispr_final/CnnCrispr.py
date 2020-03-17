import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional, LSTM, BatchNormalization
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D

VERBOSE = 1
OPTIMIZER = Adam(lr=10e-4)
VOCAB_SIZE = 16  # 4**3
EMBED_SIZE = 100
maxlen = 23  # [(L-kmer)/step] +1

def loadGlove(inputpath, outputpath=""):
    data_list = []
    wordEmb = {}
    with open(inputpath) as f:
        for line in f:
            # 基本的数据整理
            ll = line.strip().split(',')
            ll[0] = str(int(float(ll[0])))
            data_list.append(ll)

            # 构建wordembeding的选项
            ll_new = [float(i) for i in ll]
            emb = np.array(ll_new[1:], dtype="float32")
            wordEmb[str(int(ll_new[0]))] = emb

    if outputpath != "":
        with open(outputpath) as f:
            for data in data_list:
                f.writelines(' '.join(data))
        # data_list = [float(i) for i in data_list]
    return wordEmb

def CnnCrispr(model_ini):
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

def main():

    print("GloVe model loaded")
    glove_inputpath = "...\Encoded_data\Class\keras_GloVeVec_5_100_10000.csv"
    # load GloVe model
    model_glove = loadGlove(glove_inputpath)
    embedding_weights = np.zeros((VOCAB_SIZE, EMBED_SIZE))  # 词语的数量×嵌入的维度
    for i in range(VOCAB_SIZE):
        embedding_weights[i, :] = model_glove[str(i)]

    print("Building models")
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                        weights=[embedding_weights],
                        trainable=True))
    model, model_message = CnnCrispr(model)

    print("Loading weights for the models")
    model.load_weights("...\Model_save\CnnCrispr_weights.h5") # 缺少h5 的内容


    print("Loading test data")
    FILE = open("...\input_example.txt", "r")
    data = FILE.readlines()
    print(len(data))
    X_test,seq_list= test_data_read(data)
    X_test = np.array(X_test)
    print(X_test.shape)
    FILE.close()

    print("Predicting on test data")
    CnnCrispr_SCORE = model.predict(X_test, batch_size=50, verbose=0)
    print(CnnCrispr_SCORE)

    print("Saving the predition results")
    OUTPUT = open("...\output_example.txt", "w")
    OUTPUT.write("sgRNA_seq, DNA_seq, CnnCrispr_score")
    OUTPUT.write("\n")
    for l in range(len(data)):
        OUTPUT.write("\t".join(seq_list[l]))
        OUTPUT.write("\t")
        OUTPUT.write("\t".join("%s"% id for id in CnnCrispr_SCORE[l]))
        OUTPUT.write("\n")
    OUTPUT.close()


def test_data_read(lines):
    data_n = len(lines)
    data_list = []
    seq_list = []

    for l in range(data_n):
        data = lines[l].split(",")
        seq_item = data[:2]
        print(seq_item)
        data_item = [int(i) for i in data[3:]]
        print(data_item)
        data_list.append(data_item)
        seq_list.append(seq_item)

    return data_list,seq_list

if __name__ == '__main__':
    main()

