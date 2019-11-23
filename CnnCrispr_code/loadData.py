import numpy as np
import random
from collections import Counter
from keras.utils import np_utils, plot_model
import xlrd

def loadGlove(inputpath, outputpath=""):
    data_list = []
    wordEmb = {}
    with open(inputpath) as f:
        for line in f:
            ll = line.strip().split(',')
            ll[0] = str(int(float(ll[0])))
            data_list.append(ll)

            ll_new = [float(i) for i in ll]
            emb = np.array(ll_new[1:], dtype="float32")
            wordEmb[str(int(ll_new[0]))] = emb

    if outputpath != "":
        with open(outputpath) as f:
            for data in data_list:
                f.writelines(' '.join(data))
    return wordEmb

#----------------------------------------------------------------------------------------------
def loadData(inputpath):
    data_list = []
    label = []
    Negative = []
    Positive = []

    with open(inputpath) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]
            sgRNA_item = ll[0]
            DNA_item = ll[1]
            label_item = np.float(ll[2])
            data_item = [int(i) for i in ll[3:]]
            if label_item == 0.0:
                Negative.append(ll)
            else:
                Positive.append(ll)
            data_list.append(data_item)
            label.append(label_item)
    return Negative, Positive
#data_list, label,Negative_data,Positive_data,Negative_label,Positive_label,Negative_sgRNA,Positive_sgRNA,Negative_DNA,Positive_DNA

#-----------------------------------------------------------------------------------------------------------------------
ExcelFile = xlrd.open_workbook(r'...\offtarget_data\Classification\off_data_class.xlsx')
sheet = ExcelFile.sheet_by_name('sgRNA')
sgRNA_seq_list = sheet.col_values(0)

#Define the data import command for classification schema
def loadData_leave_one_out(inputpath,sgRNAList):
    data_list = []
    sgRNA_list = []

    with open(inputpath) as f:

        position_address = [[] for i in range(len(sgRNAList))]
        index = 0
        for line in f:
            ll = [i for i in line.strip().split(',')]
            sgRNA_item = ll[0]
            data_item = ll
            for i in range(len(sgRNAList)):
                if sgRNA_item == sgRNAList[i]:
                    position_address[i].append(index)
            data_list.append(data_item)
            sgRNA_list.append(sgRNA_item)
            index +=1
        position = []
        for i in range(len(sgRNAList)):
            position.append([sgRNAList[i],position_address[i]])
        dict_address = dict(position)

    return data_list,sgRNA_list,dict_address

#----------------------------------------------------------------------------------------------
#Define the data import command for regression schema
def loadData_reg_leave_one_out(inputpath):
    data_list = []
    label = []
    sgRNA_list = []

    with open(inputpath) as f:
        for line in f:
            ll = [i for i in line.strip().split(',')]
            sgRNA_item = ll[0]
            label_item = np.float(ll[1])
            data_item = [int(i) for i in ll[2:]]
            data_list.append(data_item)
            label.append(label_item)
            sgRNA_list.append(sgRNA_item)
    return data_list, label,sgRNA_list

#-----------------------------------------------------------------
# Define the data set partition operation
def Dataset_split(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices],data[test_indices]
#data_list[train_indices],data_list[test_indices],label_list[train_indices], label_list[test_indices],sgRNA_list[train_indices],sgRNA_list[test_indices],DNA_list[train_indices],DNA_list[test_indices]

#-----------------------------------------------------------------------------------------------------------------------

def make_empty_batch(batchsize):
  image_batch = np.zeros((batchsize, 23))#could change for u data
  text_batch = np.zeros((batchsize, 2))
  return image_batch, text_batch

def train_flow(Train_Negative, Train_Positive, batchsize):
    train_Negative = np.array(Train_Negative[:,3:])
    train_Positive = np.array(Train_Positive[:,3:])
    '''
    M = sorted(Counter(train_Y).items())
    N_negative = int(M[1][1])
    N_positive = int(M[0][1])
    print(N_negative)
    '''
    Num_Positive = len(train_Positive)
    Num_Negative = len(train_Negative)
    Index_Negative = [i for i in range(Num_Negative)]
    Index_Positive = np.random.randint(0, Num_Positive, batchsize,dtype='int32')
    #print(Index_Positive.shape,Index_Positive)
    random.shuffle(Index_Negative)
    Total_num_batch = int(Num_Negative / batchsize)
    batch_counter=0
    num_counter = 0
    X_input = []
    Y_input = []
    while True:
        for i in range(Total_num_batch):
            for j in range(batchsize):
                X_input.append(train_Negative[Index_Negative[j + i*batchsize]])
                Y_input.append(0)
                X_input.append(train_Positive[Index_Positive[j]])
                Y_input.append(1)
                num_counter +=1
                #print(num_counter,np.array(Y_input).shape)
                if num_counter == batchsize:
                    Y_input = np_utils.to_categorical(Y_input)
                    #print(np.array(Y_input).shape)
                    yield (np.array(X_input), np.array(Y_input))
                    X_input = []
                    Y_input = []
                    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                    num_counter = 0


def valid_flow(Test_Negative, Test_Positive, batchsize):
    valid_Negative = np.array(Test_Negative[:,3:])
    valid_Positive = np.array(Test_Positive[:,3:])

    Num_Positive = len(valid_Positive)
    Num_Negative = len(valid_Negative)
    Index_Negative = [i for i in range(Num_Negative)]
    Index_Positive = np.random.randint(0, Num_Positive, batchsize,dtype='int32')
    random.shuffle(Index_Negative)
    num_counter = 0
    X_input = []
    Y_input = []
    while True:
        for j in range(batchsize):
            X_input.append(valid_Negative[Index_Negative[j]])
            Y_input.append(0)
            X_input.append(valid_Positive[Index_Positive[j]])
            Y_input.append(1)
            num_counter +=1
            #print(num_counter,np.array(Y_input).shape)
            if num_counter == batchsize:
                Y_input = np_utils.to_categorical(Y_input)
                #print(np.array(Y_input).shape)
                yield (np.array(X_input), np.array(Y_input))
                X_input = []
                Y_input = []
                Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                num_counter = 0

#-----------------------------------------------------------------------------------------------------------------------
def train_flow_reg(Train_Negative, Train_Positive, batchsize):
    train_Negative = np.array(Train_Negative[:,3:])
    train_Positive = np.array(Train_Positive[:,3:])
    train_Negative_label = np.array([np.float(i) for i in Train_Negative[:,2]]).reshape(len(Train_Negative),1)
    train_Positive_label = np.array([np.float(i) for i in Train_Positive[:,2]]).reshape(len(Train_Positive),1)

    Num_Positive = len(train_Positive)
    Num_Negative = len(train_Negative)
    Index_Negative = [i for i in range(Num_Negative)]
    Index_Positive = np.random.randint(0, Num_Positive, batchsize,dtype='int32')
    #print(Index_Positive.shape,Index_Positive)
    random.shuffle(Index_Negative)
    Total_num_batch = int(Num_Negative / batchsize)
    batch_counter=0
    num_counter = 0
    X_input = []
    Y_input = []
    while True:
        for i in range(Total_num_batch):
            for j in range(batchsize):
                X_input.append(train_Negative[Index_Negative[j + i*batchsize]])
                Y_input.append(train_Negative_label[Index_Negative[j + i*batchsize]])
                X_input.append(train_Positive[Index_Positive[j]])
                Y_input.append(train_Positive_label[Index_Positive[j]])
                num_counter +=1
                #print(num_counter,np.array(Y_input).shape)
                if num_counter == batchsize:
                    #Y_input = np_utils.to_categorical(Y_input)
                    #print(np.array(Y_input).shape)
                    yield (np.array(X_input), np.array(Y_input))
                    X_input = []
                    Y_input = []
                    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                    num_counter = 0


def valid_flow_reg(Test_Negative, Test_Positive, batchsize):
    valid_Negative = Test_Negative[:,3:]
    valid_Positive = Test_Positive[:,3:]
    valid_Negative_label = [np.float(i) for i in Test_Negative[:,2]]
    valid_Positive_label = [np.float(i) for i in Test_Positive[:,2]]

    Num_Positive = len(valid_Positive)
    Num_Negative = len(valid_Negative)
    Index_Negative = [i for i in range(Num_Negative)]
    Index_Positive = np.random.randint(0, Num_Positive, batchsize,dtype='int32')
    random.shuffle(Index_Negative)
    num_counter = 0
    X_input = []
    Y_input = []
    while True:
        for j in range(batchsize):
            X_input.append(valid_Negative[Index_Negative[j]])
            Y_input.append(valid_Negative_label[Index_Negative[j]])
            X_input.append(valid_Positive[Index_Positive[j]])
            Y_input.append(valid_Positive_label[Index_Positive[j]])
            num_counter +=1
            #print(num_counter,np.array(Y_input).shape)
            if num_counter == batchsize:
                #Y_input = np_utils.to_categorical(Y_input)
                #print(np.array(Y_input).shape)
                yield (np.array(X_input), np.array(Y_input))
                X_input = []
                Y_input = []
                Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                num_counter = 0