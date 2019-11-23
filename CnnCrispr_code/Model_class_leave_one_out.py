from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.utils import np_utils, plot_model
from keras.optimizers import Adam, SGD
from keras.losses import logcosh, binary_crossentropy, categorical_crossentropy
from loadData import loadGlove, loadData_leave_one_out, Dataset_split, train_flow, valid_flow
from util import logTool, util
import datetime
from model_get import modelSelection
from model_eval import roc_auc, model_rep, model_matrix, specificity, sensitivity, pr_auc, roc_Curve,GetKS
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
import random
import xlrd
import pandas as pd
from CFD_score import CFD_score
from CNN_std import CNN_std_score,cnn_predict
from MITsourcecode import MIT_score

VERBOSE = 1
OPTIMIZER = Adam(lr=0.01)  # Adam(lr=10e-4)
VOCAB_SIZE = 16  # 4**3
EMBED_SIZE = 100
BATCH_SIZE = 256
TEST_SIZE = 100
NUM_EPOCHS = 30
TEST_RATIO = 0.2
maxlen = 23  # [(L-kmer)/step] +1
MODEL_select = 'Class'

#————————————————————————————————————————————-----------------------------------
# 模型及参数保存命令
model_name = "%s_%s" % (
    MODEL_select, datetime.datetime.now().strftime("%Y-%m-%d"))
# 判断路径是否存在，若不存在则创建新的目录
util.mkdir(".../keras_model/leave_one_out_Class/%s/" % model_name)

log = logTool(".../keras_model/leave_one_out_Class/%s/log.txt" % model_name)
log.info('log initiated')
np.random.seed(1671)
#-----------------------------------------------------------------------------------------------------------------------
# 数据加载及划分
glove_inputpath = "...\Data\Class_leave_one_out\keras_GloVeVec_5_100_10000.csv"
inputpath = "...\Data\Class_leave_one_out\off_Glove.txt"
# load GloVe model
model_glove = loadGlove(glove_inputpath)
embedding_weights = np.zeros((VOCAB_SIZE, EMBED_SIZE))  # Number of words x embedded dimensions
for i in range(VOCAB_SIZE):
    embedding_weights[i, :] = model_glove[str(i)]
print('GloVe model loaded')
log.info("GloVe model loaded")

ExcelFile = xlrd.open_workbook(r'...\offtarget_data\Classification\off_data_class.xlsx')
sheet = ExcelFile.sheet_by_name('sgRNA')
sgRNA_seq_list = sheet.col_values(0)
print(np.array(sgRNA_seq_list).shape)
data, sgRNA_list,dict_address = loadData_leave_one_out(inputpath,sgRNA_seq_list)

positoin_address = []
for i in sgRNA_list:
    address_index = [x for x in range(len(sgRNA_list)) if sgRNA_list[x] == i]
    positoin_address.append([i, address_index])
dict_address = dict(positoin_address)

keys = dict_address.keys()

ROC_Mean = [[0 for i in range(3)] for j in range(len(keys))]
PRC_Mean = [[0 for i in range(3)] for j in range(len(keys))]
Pearson_Mean = [[0 for i in range(3)] for j in range(len(keys))]
Spearman_Mean = [[0 for i in range(3)] for j in range(len(keys))]
sgRNA_num = 0
for key in keys:
    sgRNA_num = sgRNA_num+1
    print("Training for the %sth time"%sgRNA_num)
    print("Leave-one-sgRNA-out:", key)
    log.info("Training for the %sth time"%sgRNA_num)
    log.info("Leave-one-sgRNA-out:%s"%key)
    test_index = dict_address[key]
    Negative_Xtest = []
    Positive_Xtest = []
    Negative_Xtrain = []
    Positive_Xtrain = []
    for i in range(len(sgRNA_list)):
        if i in test_index:
            if np.float(data[i][2]) > 0.0:
                Positive_Xtest.append(data[i])
            else:
                Negative_Xtest.append(data[i])
        else:
            if np.float(data[i][2]) > 0.0:
                Positive_Xtrain.append(data[i])
            else:
                Negative_Xtrain.append(data[i])

    Xtest = np.vstack((Negative_Xtest,Positive_Xtest))
    Xtest_Seq = [i for i in range(len(Xtest))]
    random.shuffle(Xtest_Seq)
    Xtest = Xtest[Xtest_Seq]
    Negative_Xtrain = np.array(Negative_Xtrain)
    Positive_Xtrain = np.array(Positive_Xtrain)
    Positive_Xtest = np.array(Positive_Xtest)
    Negative_Xtest = np.array(Negative_Xtest)
    Xtest = np.array(Xtest)
    print("Data amount of training set: %s, data amount of test set: %s"%(len(Positive_Xtrain)+len(Negative_Xtrain),len(Xtest)))
    log.info("Data amount of training set: %s, data amount of test set: %s" % (len(Positive_Xtrain) + len(Negative_Xtrain), len(Xtest)))
    NUM_BATCH = int(len(Negative_Xtrain) / BATCH_SIZE)

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                        weights=[embedding_weights],
                        trainable=True))
    model, model_message = modelSelection(model, MODEL_select)
    model.summary()
    log.info("model loaded")
    print("model loaded")
    log.info(model_message)
    print(model_message)
    model.compile(loss=binary_crossentropy,
                  optimizer=OPTIMIZER, metrics=['acc', roc_auc])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5)
    History = model.fit_generator(train_flow(Negative_Xtrain, Positive_Xtrain, BATCH_SIZE), shuffle=True,
                                  validation_data=valid_flow(Negative_Xtest, Positive_Xtest, TEST_SIZE),
                                  validation_steps=1,
                                  steps_per_epoch=NUM_BATCH, epochs=NUM_EPOCHS, verbose=VERBOSE, callbacks=[reduce_lr])

    log.info("finish training")
    Ypred = model.predict(Xtest[:,3:])
    # print(Ypred)
    Y_score = Ypred[:, 1]
    Y_score = np.array(Y_score)
    print(Y_score.shape)
    score = model.evaluate(Xtest[:,3:], Ytest, verbose=0)
    Ypred = [int(i[1] > i[0]) for i in Ypred]

    Ytest = [1 if np.float(i) > 0.0 else 0 for i in Xtest[:, 2]]
    Ytest = np_utils.to_categorical(Ytest)
    # Y_train_pred = model.predict()

    Ytest = [int(i[1] > i[0]) for i in Ytest]

    plt.figure(figsize=(15, 7))
    roc_value, prc_value, ks = GetKS(Ytest, Y_score)
    print("CnnCrispr: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f" % (roc_value, prc_value, ks))
    log.info("CnnCrispr: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f" % (roc_value, prc_value, ks))


    Ytest = np.array(Ytest)
    Ytest = Ytest.reshape(len(Ytest), 1)

    p, r, f1score = model_rep(Ytest, Ypred)
    conf_matric = model_matrix(Ytest, Ypred)
    print("Test score: {:.3f}, accuracy: {:.3f}, auc: {:.3f}".format(
        score[0], score[1], score[2]))
    log.info("Test score: {:.3f}, accuracy: {:.3f}, auc: {:.3f}".format(
        score[0], score[1], score[2]))
    print(
        "The result of Test_set: precision: {:.3f}, recall: {:.3f}, F1-score: {:.3f}".format(p, r, f1score))
    log.info(
        "precision: {:.3f}, recall: {:.3f}, F1-score: {:.3f}".format(p, r, f1score))
    print("The number of positive data in the total test set is %s" % (len(Positive_Xtest)))
    print("confusion matric:\n %s" % (conf_matric))

    log.info("confusion matric:\n %s" % (conf_matric))
    ROC_Mean[sgRNA_num-1][0]=roc_value
    PRC_Mean[sgRNA_num-1][0]=prc_value

    CFD_value = CFD_score(Xtest[:, 0], Xtest[:, 1])
    CFD_value = np.array(CFD_value)

    roc_value, prc_value, ks = GetKS(Ytest, CFD_value, label="CFD")
    #print(CFD_value.shape,Ytest.shape)
    CFD_value = CFD_value.reshape(len(CFD_value), 1)
    CFD_data = np.hstack((Ytest, CFD_value))
    CFD_df = pd.DataFrame(CFD_data, columns=['Ytest', 'Ypred'])
    print("CFD: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f" % (roc_value, prc_value, ks))
    print("Pearson_value:\n", '\n', CFD_df.corr('pearson'), '\n', CFD_df.corr('spearman'))

    log.info("CFD: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f" % (roc_value, prc_value, ks))
    ROC_Mean[sgRNA_num - 1][1] = roc_value
    PRC_Mean[sgRNA_num - 1][1] = prc_value
    CNN_std_value = cnn_predict(Xtest)
    CNN_std_value = np.array(CNN_std_value)
    roc_value, prc_value, ks = GetKS(Ytest, CNN_std_value, label="CNN_std")
    plt.tight_layout()
    plt.savefig("D:\ProcessAndData\Glove_CnnCrispr\keras_model\Figures\%s_No%s.png"%(key,sgRNA_num))
    plt.close()
    print("CNN_std: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f" % (roc_value, prc_value, ks))
    log.info("CNN_std: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f" % (roc_value, prc_value, ks))
    ROC_Mean[sgRNA_num - 1][2] = roc_value
    PRC_Mean[sgRNA_num - 1][2] = prc_value

    MIT_value = MIT_score(Xtest[:, 0], Xtest[:, 1])
    MIT_value = np.array(MIT_value)
    roc_value, prc_value, ks = GetKS(Ytest, MIT_value, label="MIT")
    print("MIT: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f" % (roc_value, prc_value, ks))
    log.info("MIT: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f" % (roc_value, prc_value, ks))
    ROC_Mean[sgRNA_num - 1][3] = roc_value
    PRC_Mean[sgRNA_num - 1][3] = prc_value
    model.save(".../keras_model/leave_one_out_Class/%s/model_%s.h5" %
               (model_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    log.info("model saved: %s_%s" % (model_name, datetime.datetime.now()))


ROC_Mean = np.array(ROC_Mean)
PRC_Mean = np.array(PRC_Mean)
print("The result of cross validation under Leave_one_sgRNA_out：")
print("ROC_Mean=%0.3f" % (np.mean(ROC_Mean)))
print("PRC_Mean=%0.3f",np.mean(PRC_Mean))
log.info("The result of cross validation under Leave_one_sgRNA_out：")
log.info("ROC_Mean=%0.3f" % (np.mean(ROC_Mean)))
log.info("PRC_Mean=%0.3f"%np.mean(PRC_Mean))
log.info("The result of cross validation under Leave_one_sgRNA_out：")
log.info("ROC_value=%s"%ROC_Mean)
log.info("PRC_value=%s"%PRC_Mean)
plt.boxplot(ROC_Mean)
plt.title("ROC_AUC")
plt.ylim([-0.05,1.2])
plt.show()
plt.boxplot(PRC_Mean)
plt.title("PRC_AUC")
plt.ylim([-0.05,1.2])
plt.show()

