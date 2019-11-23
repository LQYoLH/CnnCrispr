from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.losses import  binary_crossentropy
from loadData import loadGlove, loadData, Dataset_split, train_flow, valid_flow
from util import logTool, util
import datetime
from model_get import modelSelection
from model_eval import roc_auc, model_rep, model_matrix, GetKS
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
plt.rc('font',family='Times New Roman')
import random
from CFD_score import CFD_score
from CNN_std import cnn_predict,CNN_std_score
from MITsourcecode import MIT_score


VERBOSE = 1
OPTIMIZER = Adam(lr=0.01)  # Adam(lr=10e-4)
VOCAB_SIZE = 16  # 4**3
EMBED_SIZE = 100
BATCH_SIZE = 256
TEST_SIZE = 100
NUM_EPOCHS = 20
TEST_RATIO = 0.2
maxlen = 23  # [(L-kmer)/step] +1
MODEL_select = 'Class_Conv_LSTM'
class_weight = dict({1: 1, 0: 250})

#————————————————————————————————————————————-----------------------------------
model_name = "%s_%s" % (
    MODEL_select, datetime.datetime.now().strftime("%Y-%m-%d"))

util.mkdir(".../keras_model/Class/%s/" % model_name)

log = logTool(".../keras_model/Class/%s/log.txt" % model_name)
log.info('log initiated')
np.random.seed(1671)

#-----------------------------------------------------------------------------------------------------------------------
glove_inputpath = "...\Data\Class\keras_GloVeVec_5_100_10000.csv"
hek_inputpath = "...\Data\Class\hek293_off_Glove.txt"
K562_inputpath = "...\Data\Class\K562_off_Glove.txt"
# load GloVe model
model_glove = loadGlove(glove_inputpath)
embedding_weights = np.zeros((VOCAB_SIZE, EMBED_SIZE))
for i in range(VOCAB_SIZE):
    embedding_weights[i, :] = model_glove[str(i)]
print('GloVe model loaded')
log.info("GloVe model loaded")

# loadData
hek_Negative, hek_Positive = loadData(hek_inputpath)
K562_Negative, K562_Positive = loadData(K562_inputpath)


hek_Negative = np.array(hek_Negative)
hek_Positive = np.array(hek_Positive)
K562_Negative = np.array(K562_Negative)
K562_Positive = np.array(K562_Positive)
hek_Train_Negative,hek_Test_Negative = Dataset_split(hek_Negative,TEST_RATIO)
hek_Train_Positive, hek_Test_Positive = Dataset_split(hek_Positive,TEST_RATIO)
K562_Train_Negative, K562_Test_Negative = Dataset_split(K562_Negative,TEST_RATIO)
K562_Train_Positive, K562_Test_Positive = Dataset_split(K562_Positive,TEST_RATIO)

Train_Negative = np.vstack((hek_Train_Negative, K562_Train_Negative))
Train_Positive = np.vstack((hek_Train_Positive, K562_Train_Positive))
print(np.array(Train_Negative).shape, np.array(Train_Positive).shape)
NUM_BATCH = int(len(Train_Negative) / BATCH_SIZE)

hek_test_size = len(hek_Test_Positive)
K562_test_size = len(K562_Test_Positive)

hek_Seq_N = [i for i in range(len(hek_Test_Negative))]
random.shuffle(hek_Seq_N)
hek_Seq_P = np.random.randint(0, len(hek_Test_Positive), hek_test_size)
K562_Seq_N = [i for i in range(len(K562_Test_Negative))]
random.shuffle(K562_Seq_N)
K562_Seq_P = np.random.randint(0, len(K562_Test_Positive), K562_test_size)
Test_Positive = np.vstack((hek_Test_Positive, K562_Test_Positive))
Test_Negative = np.vstack((hek_Test_Negative, K562_Test_Negative))
hek_Xtest = np.vstack((hek_Test_Negative,hek_Test_Positive))
K562_Xtest = np.vstack((K562_Test_Negative,K562_Test_Positive))
Xtest = np.vstack((hek_Xtest,K562_Xtest))
hek_Seq = [i for i in range(len(hek_Xtest))]
random.shuffle(hek_Seq)
K562_Seq = [i for i in range(len(K562_Xtest))]
random.shuffle(K562_Seq)
Xtest_Seq = [i for i in range(len(Xtest))]
random.shuffle(Xtest_Seq)
hek_Xtest = hek_Xtest[hek_Seq]
K562_Xtest = K562_Xtest[K562_Seq]
Xtest = Xtest[Xtest_Seq]

Xtest = np.array(Xtest)
hek_Xtest = np.array(hek_Xtest)
K562_Xtest = np.array(K562_Xtest)
print(hek_Xtest.shape)
print(K562_Xtest.shape)
print(Xtest.shape)

#-----------------------------------------------------------------------------------------------------------------------
Ytest = [1 if np.float(i) >0.0 else 0 for i in Xtest[:,2]]
Ytest = np_utils.to_categorical(Ytest)
sgRNA = np.array(Xtest[:,0])
DNA_list = np.array(Xtest[:,1])
#print(sgRNA.shape,sgRNA[1:10])
CFD_value = CFD_score(Xtest[:,0],Xtest[:,1])
CFD_value = np.array(CFD_value)
#print(max(CFD_value))

CRISPOR_value = cnn_predict(Xtest)
CRISPOR_value = np.array(CRISPOR_value)
print(CRISPOR_value.shape)

MIT_value = MIT_score(Xtest[:,0],Xtest[:,1])
MIT_value = np.array(MIT_value)

#CNN_std_value = CNN_std_score(Xtest[:,0],Xtest[:,1])
#CNN_std_value = np.array(CNN_std_value)
print(MIT_value.shape)
print(MIT_value[1:10])
print(CFD_value[1:10])
#print(CNN_std_value[1:10])


hek_Ytest = [1 if np.float(i) >0.0 else 0 for i in hek_Xtest[:,2]]
hek_Ytest = np_utils.to_categorical(hek_Ytest)
hek_sgRNA = np.array(hek_Xtest[:,0])
hek_DNA_list = np.array(hek_Xtest[:,1])
#print(sgRNA.shape,sgRNA[1:10])
hek_CFD_value = CFD_score(hek_sgRNA,hek_DNA_list)
hek_CFD_value = np.array(hek_CFD_value)
print(max(hek_CFD_value))
hek_CRISPOR_value = cnn_predict(hek_Xtest)
hek_CRISPOR_value = np.array(hek_CRISPOR_value)
print(hek_CRISPOR_value.shape)
hek_MIT_value = MIT_score(hek_sgRNA,hek_DNA_list)
hek_MIT_value = np.array(hek_MIT_value)


K562_Ytest = [1 if np.float(i) >0.0 else 0 for i in K562_Xtest[:,2]]
K562_Ytest = np_utils.to_categorical(K562_Ytest)
K562_sgRNA = np.array(K562_Xtest[:,0])
K562_DNA_list = np.array(K562_Xtest[:,1])
#print(sgRNA.shape,sgRNA[1:10])
K562_CFD_value = CFD_score(K562_sgRNA,K562_DNA_list)
K562_CFD_value = np.array(K562_CFD_value)
print(max(K562_CFD_value))
K562_CRISPOR_value = cnn_predict(K562_Xtest)
K562_CRISPOR_value = np.array(K562_CRISPOR_value)
print(K562_CRISPOR_value.shape)

K562_MIT_value = MIT_score(K562_sgRNA,K562_DNA_list)
K562_MIT_value = np.array(K562_MIT_value)


#-----------------------------------------------------------------------------------------------------------------------
# Model training and performance evaluation
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
# plot_model(model,to_file="model4.png",show_shapes=True)
# plot_model(model, to_file="../../Data/keras_model/%s/model.png" %
#            model_name, show_shapes=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5)
History = model.fit_generator(train_flow(Train_Negative, Train_Positive, BATCH_SIZE), shuffle=True,validation_data=valid_flow(Test_Negative,Test_Positive,TEST_SIZE),validation_steps=1,
                              steps_per_epoch=NUM_BATCH, epochs=NUM_EPOCHS, verbose=VERBOSE, callbacks=[reduce_lr])

print(History.history.keys())
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend(['train','test'],loc='upper left')
plt.show()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend(['train','test'],loc='upper left')
plt.show()
plt.plot(History.history['roc_auc'])
plt.plot(History.history['val_roc_auc'])
plt.title("model auc")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend(['train','test'],loc='upper left')
plt.show()
plt.plot(History.history['lr'])
plt.show()

log.info("finish training")

print("The number of positive data in total test set was %s"%(len(hek_Test_Positive)+len(K562_Test_Positive)))
Ypred = model.predict(Xtest[:,3:])
#print(Ypred)
Y_score = Ypred[:, 1]
Y_score = np.array(Y_score)
print(Y_score.shape)
# Y_train_pred = model.predict()
score = model.evaluate(Xtest[:,3:], Ytest, verbose=0)
Ypred = [int(i[1] > i[0]) for i in Ypred]
Ytest = [int(i[1] > i[0]) for i in Ytest]
plt.figure(figsize=(15, 7))
roc_value,prc_value, ks = GetKS(Ytest,Y_score)
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(Ytest,CFD_value,label="CFD")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(Ytest,MIT_value,label="MIT")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(Ytest,CRISPOR_value,label="CNN_std")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
plt.tight_layout()
plt.savefig(".../keras_model/Class/%s/Total_set.png" %(model_name))
plt.show()
plt.close()

Ytest = np.array(Ytest)
Ytest = Ytest.reshape(len(Ytest), 1)
p, r, f1score = model_rep(Ytest, Ypred)
#CFD_Ypred = [round(i) for i in CFD_value]
#p1,r1,f1 = model_rep(Ytest, CFD_Ypred)
#print(r1)
print(Ytest.shape,MIT_value.shape,CFD_value.shape,CRISPOR_value.shape,np.array(Ypred).shape)
MIT_Ypred = [round(i) for i in MIT_value]
CFD_Ypred = [round(i) for i in CFD_value]
CNN_std_Ypred = [np.round(i) for i in CRISPOR_value]
conf_matrix = model_matrix(Ytest, Ypred)
MIT_matrix = model_matrix(Ytest,MIT_Ypred)
CFD_matrix = model_matrix(Ytest,CFD_Ypred)
CRISPOR_matrix = model_matrix(Ytest,CNN_std_Ypred)
print("Test score: {:.3f}, accuracy: {:.3f}, auc: {:.3f}".format(
    score[0], score[1], score[2]))
log.info("Test score: {:.3f}, accuracy: {:.3f}, auc: {:.3f}".format(
    score[0], score[1], score[2]))
print(
    "The result of Test_set: precision: {:.3f}, recall: {:.3f}, F1-score: {:.3f}".format(p, r, f1score))
log.info(
    "precision: {:.3f}, recall: {:.3f}, F1-score: {:.3f}".format(p, r, f1score))
print("confusion matric:\n %s" % (conf_matrix))

print("MIT matric:\n %s"%MIT_matrix)
print("CFD matric:\n %s"%CFD_matrix)
print("CNN_std matric:\n %s"%CRISPOR_matrix)

log.info("confusion matric:\n %s" % (conf_matrix))
log.info("MIT matric:\n %s"%MIT_matrix)
log.info("CFD matric:\n %s"%CFD_matrix)
log.info("CNN_std matric:\n %s"%CRISPOR_matrix)

print("The number of positive data in Hek293t test set was %ss"%(len(hek_Test_Positive)))
hek_Ypred = model.predict(hek_Xtest[:,3:])
#print(hek_Ypred)
hek_Y_score = hek_Ypred[:, 1]
hek_Y_score = np.array(hek_Y_score)
print(hek_Y_score.shape)
# Y_train_pred = model.predict()
score = model.evaluate(hek_Xtest[:,3:], hek_Ytest, verbose=0)
hek_Ypred = [int(i[1] > i[0]) for i in hek_Ypred]
hek_Ytest = [int(i[1] > i[0]) for i in hek_Ytest]
plt.figure(figsize=(15, 7))
roc_value,prc_value, ks = GetKS(hek_Ytest,hek_Y_score)
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(hek_Ytest,hek_CFD_value,label="CFD")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(hek_Ytest,hek_MIT_value,label='MIT')
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(hek_Ytest,hek_CRISPOR_value,label='CNN_std')
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
plt.tight_layout()
plt.savefig(".../keras_model/Class/%s/hek293t_set.png" %(model_name))
plt.show()
plt.close()

MIT_Ypred = [round(i) for i in hek_MIT_value]
CFD_Ypred = [round(i) for i in hek_CFD_value]
CNN_std_Ypred = [np.round(i) for i in hek_CRISPOR_value]

hek_Ytest = np.array(hek_Ytest)
hek_Ytest = hek_Ytest.reshape(len(hek_Ytest), 1)
p, r, f1score = model_rep(hek_Ytest, hek_Ypred)
conf_matric = model_matrix(hek_Ytest, hek_Ypred)


CFD_matric = model_matrix(hek_Ytest, CFD_Ypred)
MIT_matric = model_matrix(hek_Ytest, MIT_Ypred)
CRISPOR_matric = model_matrix(hek_Ytest, CNN_std_Ypred)


print("hek_Test score: {:.3f}, accuracy: {:.3f}, auc: {:.3f}".format(
    score[0], score[1], score[2]))
log.info("hek_Test score: {:.3f}, accuracy: {:.3f}, auc: {:.3f}".format(
    score[0], score[1], score[2]))
print(
    "The result of hek_Test_set: precision: {:.3f}, recall: {:.3f}, F1-score: {:.3f}".format(p, r, f1score))
log.info(
    "hek_precision: {:.3f}, recall: {:.3f}, F1-score: {:.3f}".format(p, r, f1score))
print("hek_confusion matric:\n %s" % (conf_matric))
log.info("confusion matric:\n %s" % (conf_matric))

print("hek_CFD_confusion matric:\n %s" % (CFD_matric))
print("hek_MIT_confusion matric:\n %s" % (MIT_matric))
print("hek_CRISPOR_confusion matric:\n %s" % (CRISPOR_matric))

log.info("hek_CFD_confusion matric:\n %s" % (CFD_matric))
log.info("hek_MIT_confusion matric:\n %s" % (MIT_matric))
log.info("hek_CRISPOR_confusion matric:\n %s" % (CRISPOR_matric))


print("The number of positive data in K562 test set was %s"%(len(K562_Test_Positive)))
K562_Ypred = model.predict(K562_Xtest[:,3:])
#print(K562_Ypred)
K562_Y_score = K562_Ypred[:, 1]
K562_Y_score = np.array(K562_Y_score)
print(K562_Y_score.shape)
# Y_train_pred = model.predict()
score = model.evaluate(K562_Xtest[:,3:], K562_Ytest, verbose=0)
K562_Ypred = [int(i[1] > i[0]) for i in K562_Ypred]
K562_Ytest = [int(i[1] > i[0]) for i in K562_Ytest]
plt.figure(figsize=(15, 7))
roc_value,prc_value, ks = GetKS(K562_Ytest,K562_Y_score,label="CnnCrispr")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(K562_Ytest,K562_CFD_value,label="CFD")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(K562_Ytest,K562_MIT_value,label="MIT")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(K562_Ytest,K562_CRISPOR_value,label="CNN_std")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))


plt.tight_layout()
plt.savefig(".../keras_model/Class/%s/K562_set.png" %(model_name))
plt.show()
plt.close()


MIT_Ypred = [round(i) for i in K562_MIT_value]
CFD_Ypred = [round(i) for i in K562_CFD_value]
CNN_std_Ypred = [np.round(i) for i in K562_CRISPOR_value]

K562_Ytest = np.array(K562_Ytest)
K562_Ytest = K562_Ytest.reshape(len(K562_Ytest), 1)
p, r, f1score = model_rep(K562_Ytest, K562_Ypred)
conf_matric = model_matrix(K562_Ytest, K562_Ypred)

CFD_matric = model_matrix(K562_Ytest, CFD_Ypred)
MIT_matric = model_matrix(K562_Ytest, MIT_Ypred)
CRISPOR_matric = model_matrix(K562_Ytest, CNN_std_Ypred)

print("Test score: {:.3f}, accuracy: {:.3f}, auc: {:.3f}".format(
    score[0], score[1], score[2]))
log.info("Test score: {:.3f}, accuracy: {:.3f}, auc: {:.3f}".format(
    score[0], score[1], score[2]))
print(
    "The result of Test_set: precision: {:.3f}, recall: {:.3f}, F1-score: {:.3f}".format(p, r, f1score))
log.info(
    "precision: {:.3f}, recall: {:.3f}, F1-score: {:.3f}".format(p, r, f1score))
print("confusion matric:\n %s" % (conf_matric))
log.info("confusion matric:\n %s" % (conf_matric))

print("CFD confusion matric:\n %s" % (CFD_matric))
print("MIT confusion matric:\n %s" % (MIT_matric))
print("CNN_std confusion matric:\n %s" % (CRISPOR_matric))

log.info("CFD confusion matric:\n %s" % (CFD_matric))
log.info("MIT confusion matric:\n %s" % (MIT_matric))
log.info("CNN_std confusion matric:\n %s" % (CRISPOR_matric))


model.save(".../keras_model/Class/%s/model_%s.h5" %
           (model_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
model.save_weights('.../keras_model/Class/%s/model_weights_%s.h5'% (model_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
log.info("model saved: %s_%s" % (model_name, datetime.datetime.now()))