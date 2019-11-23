from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from loadData import loadGlove, loadData, Dataset_split, train_flow_reg, valid_flow_reg
from util import logTool, util
import datetime
from model_get import modelSelection
from model_eval import roc_auc, model_rep, model_matrix, specificity, sensitivity, pr_auc, roc_Curve,GetKS
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
from model_performance import Roc_curve, pearson_r, performance, roc_auc_curve, spearman_co
import random
import re
import pickle
import pandas as pd
from CFD_score import CFD_score
from CNN_std import cnn_predict
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
MODEL_select = 'Reg'
class_weight = dict({1: 1, 0: 250})
#————————————————————————————————————————————-----------------------------------
# 模型及参数保存命令
model_name = "%s_%s" % (
    MODEL_select, datetime.datetime.now().strftime("%Y-%m-%d"))

util.mkdir(".../keras_model/Reg/%s/" % model_name)

log = logTool(".../keras_model/Reg/%s/log.txt" % model_name)
log.info('log initiated')
np.random.seed(1671)

#-----------------------------------------------------------------------------------------------------------------------

glove_inputpath = "...\Data\Reg\keras_GloVeVec_5_100_10000.csv"
hek_inputpath = "...\Data\Reg\hek293_off_Glove.txt"
K562_inputpath = "...\Data\Reg\K562_off_Glove.txt"
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
print("Train_Negative:%s, Train_Positive:%s"%(np.array(Train_Negative).shape, np.array(Train_Positive).shape))
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

Ytest = [np.float(i) for i in Xtest[:,2]]
Y_test = [1 if np.float(i)>0.0 else 0 for i in Xtest[:,2]]
Ytest = np.array(Ytest).reshape(len(Ytest),1)
Y_test = np.array(Y_test).reshape(len(Y_test),1)
sgRNA = np.array(Xtest[:,0])
DNA_list = np.array(Xtest[:,1])

CFD_value = CFD_score(Xtest[:,0],Xtest[:,1])
CFD_value = np.array(CFD_value)
print(max(CFD_value))
CRISPOR_value = cnn_predict(Xtest)
CRISPOR_value = np.array(CRISPOR_value)
print(CRISPOR_value.shape)

MIT_value = MIT_score(Xtest[:,0],Xtest[:,1])
MIT_value = np.array(MIT_value)

hek_Ytest = [np.float(i) for i in hek_Xtest[:,2]]
hek_Y_test = [1 if np.float(i)>0.0 else 0 for i in hek_Xtest[:,2]]
hek_Ytest = np.array(hek_Ytest).reshape(len(hek_Ytest),1)
hek_Y_test = np.array(hek_Y_test).reshape(len(hek_Y_test),1)
hek_sgRNA = np.array(hek_Xtest[:,0])
hek_DNA_list = np.array(hek_Xtest[:,1])
#print(sgRNA.shape,sgRNA[1:10])
hek_CFD_value = CFD_score(hek_Xtest[:,0],hek_Xtest[:,1])
hek_CFD_value = np.array(hek_CFD_value)
print(max(hek_CFD_value))
hek_CRISPOR_value = cnn_predict(hek_Xtest)
hek_CRISPOR_value = np.array(hek_CRISPOR_value)
print(hek_CRISPOR_value.shape)
hek_MIT_value = MIT_score(hek_sgRNA,hek_DNA_list)
hek_MIT_value = np.array(hek_MIT_value)

K562_Ytest = [np.float(i) for i in K562_Xtest[:,2]]
K562_Y_test = [1 if np.float(i)>0.0 else 0 for i in K562_Xtest[:,2]]
K562_Ytest = np.array(K562_Ytest).reshape(len(K562_Ytest),1)
K562_Y_test = np.array(K562_Y_test).reshape(len(K562_Y_test),1)

#print(sgRNA.shape,sgRNA[1:10])
K562_CFD_value = CFD_score(K562_Xtest[:,0],K562_Xtest[:,1])
K562_CFD_value = np.array(K562_CFD_value)
print(max(K562_CFD_value))
K562_CRISPOR_value = cnn_predict(K562_Xtest)
K562_CRISPOR_value = np.array(K562_CRISPOR_value)
print(K562_CRISPOR_value.shape)
K562_MIT_value = MIT_score(K562_Xtest[:,0],K562_Xtest[:,1])
K562_MIT_value = np.array(K562_MIT_value)
# print(X)
# print(Y)

#-----------------------------------------------------------------------------------------------------------------------

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
model.compile(loss='mse',
              optimizer=OPTIMIZER, metrics=['mae'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5)
History = model.fit_generator(train_flow_reg(Train_Negative, Train_Positive, BATCH_SIZE), shuffle=True,validation_data=valid_flow_reg(Test_Negative,Test_Positive, TEST_SIZE),validation_steps=1,
                              steps_per_epoch=NUM_BATCH, epochs=NUM_EPOCHS, verbose=VERBOSE, callbacks=[reduce_lr])

print(History.history.keys())
plt.plot(History.history['mean_absolute_error'])
plt.plot(History.history['val_mean_absolute_error'])
plt.title("model mae")
plt.xlabel("epoch")
plt.ylabel("MAE")
plt.legend(['train','test'],loc='upper left')
plt.show()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend(['train','test'],loc='upper left')
plt.show()
plt.plot(History.history['lr'])
plt.show()

log.info("finish training")
#-----------------------------------------------------------------------------------------------------------------------
print("The number of positive data in the total test set is %s"%(len(hek_Test_Positive)+len(K562_Test_Positive)))
Ypred = model.predict(Xtest[:,3:])
print(type(Ypred),Ypred.shape)
Y_score = Ypred

X_test = np.array(Xtest[:,3:])
# Y_train_pred = model.predict()
score = model.evaluate(Xtest[:,3:], Ytest, verbose=0)

df_data = np.hstack((Ytest,Ypred))
df = pd.DataFrame(df_data,columns=['Ytest','Ypred'])
CFD_value = CFD_value.reshape(len(CFD_value),1)
MIT_value = MIT_value.reshape(len(MIT_value),1)
CFD_data = np.hstack((Ytest,CFD_value))
CFD_df = pd.DataFrame(CFD_data,columns=['Ytest','Ypred'])
CRISPOR_data = np.hstack((Ytest,CRISPOR_value))
CRISPOR_df = pd.DataFrame(CRISPOR_data,columns=['Ytest','Ypred'])
MIT_data = np.hstack((Ytest,MIT_value))
MIT_df = pd.DataFrame(MIT_data,columns=['Ytest','Ypred'])
print("Pearson_value:\n",'\n',df.corr('pearson'),'\n',CFD_df.corr('pearson'),'\n',CRISPOR_df.corr('pearson'),'\n',MIT_df.corr('pearson'))
print("Spearman_value:\n",'\n',df.corr('spearman'),'\n',CFD_df.corr('spearman'),'\n',CRISPOR_df.corr('spearman'),'\n',MIT_df.corr('spearman'))
log.info("Pearson value: CnnCrispr vs CFD vs CNN_std vs MIT")
log.info(df.corr('pearson'))
log.info(CFD_df.corr('pearson'))
log.info(CRISPOR_df.corr('pearson'))
log.info(MIT_df.corr('pearson'))
log.info("Spearman value: CnnCrispr vs CFD vs CNN_std vs MIT")
log.info(df.corr('spearman'))
log.info(CFD_df.corr('spearman'))
log.info(CRISPOR_df.corr('spearman'))
log.info(MIT_df.corr('spearman'))

plt.figure(figsize=(15, 7))
roc_value,prc_value, ks = GetKS(Y_test,Y_score,label="CnnCrispr")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(Y_test,CFD_value,label="CFD")
print("CFD result: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(Y_test,CRISPOR_value,label="CNN_std")
print("CNN_std result:ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
plt.tight_layout()
plt.savefig(".../keras_model/Reg/%s/Total_set.png"%model_name)
plt.show()
plt.close()


#-----------------------------------------------------------------------------------------------------------------------
print("The number of positive data in Hek293t test set is N%s"%(len(hek_Test_Positive)))
hek_Ypred = model.predict(hek_Xtest[:,3:])
hek_Y_score = hek_Ypred
hek_Y_score = np.array(hek_Y_score)
# Y_train_pred = model.predict()
score = model.evaluate(hek_Xtest[:,3:], hek_Ytest, verbose=0)

df_data = np.hstack((hek_Ytest, hek_Ypred))
df = pd.DataFrame(df_data,columns=['Ytest','Ypred'])
hek_CFD_value = hek_CFD_value.reshape(len(hek_CFD_value),1)
hek_MIT_value = hek_MIT_value.reshape(len(hek_MIT_value),1)
CFD_data = np.hstack((hek_Ytest, hek_CFD_value))
CFD_df = pd.DataFrame(CFD_data,columns=['Ytest','Ypred'])
CRISPOR_data = np.hstack((hek_Ytest, hek_CRISPOR_value))
CRISPOR_df = pd.DataFrame(CRISPOR_data,columns=['Ytest','Ypred'])
MIT_data = np.hstack((hek_Ytest, hek_MIT_value))
MIT_df = pd.DataFrame(MIT_data,columns=['Ytest','Ypred'])
print("Pearson_value in Hek Test set :\n",'\n',df.corr('pearson'),'\n',CFD_df.corr('pearson'),'\n',CRISPOR_df.corr('pearson'),'\n',MIT_df.corr('pearson'))
print("Spearman_value in Hek Test set :\n",'\n',df.corr('spearman'),'\n',CFD_df.corr('spearman'),'\n',CRISPOR_df.corr('spearman'),'\n',MIT_df.corr('spearman'))
log.info("Pearson value in Hek test set: CnnCrispr vs CFD vs CNN_std vs MIT")
log.info(df.corr('pearson'))
log.info(CFD_df.corr('pearson'))
log.info(CRISPOR_df.corr('pearson'))
log.info(MIT_df.corr('pearson'))
log.info("Spearman value in Hek test set: CnnCrispr vs CFD vs CNN_std vs MIT")
log.info(df.corr('spearman'))
log.info(CFD_df.corr('spearman'))
log.info(CRISPOR_df.corr('spearman'))
log.info(MIT_df.corr('spearman'))

plt.figure(figsize=(15, 7))
roc_value,prc_value, ks = GetKS(hek_Y_test,hek_Y_score,label="CnnCrispr")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("CnnCrispr:ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(hek_Y_test,hek_CFD_value,label="CFD")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("CFD:ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(hek_Y_test,hek_CRISPOR_value,label="CNN_std")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("CNN_std: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
plt.tight_layout()
plt.savefig(".../keras_model/Reg/%s/Hek293t_set.png"%model_name)
plt.show()
plt.close()

#-----------------------------------------------------------------------------------------------------------------------
#K562测试集测试结果
print("The number of positive data in K562 test set is %s"%(len(K562_Test_Positive)))
K562_Ypred = model.predict(K562_Xtest[:,3:])
K562_Y_score = K562_Ypred
K562_Y_score = np.array(K562_Y_score)
score = model.evaluate(K562_Xtest[:,3:], K562_Ytest, verbose=0)

df_data = np.hstack((K562_Ytest,K562_Ypred))
df = pd.DataFrame(df_data,columns=['Ytest','Ypred'])
K562_CFD_value = K562_CFD_value.reshape(len(K562_CFD_value),1)
K562_MIT_value = K562_MIT_value.reshape(len(K562_MIT_value),1)
CFD_data = np.hstack((K562_Ytest,K562_CFD_value))
CFD_df = pd.DataFrame(CFD_data,columns=['Ytest','Ypred'])
CRISPOR_data = np.hstack((K562_Ytest,K562_CRISPOR_value))
CRISPOR_df = pd.DataFrame(CRISPOR_data,columns=['Ytest','Ypred'])
MIT_data = np.hstack((K562_Ytest,K562_MIT_value))
MIT_df = pd.DataFrame(MIT_data,columns=['Ytest','Ypred'])
print("Pearson_value in K562 test set:\n",df.corr('pearson'),'\n',CFD_df.corr('pearson'),'\n',CRISPOR_df.corr('pearson'),'\n',MIT_df.corr('pearson'))
print("Spearman_value in K562 test set:\n",df.corr('spearman'),'\n',CFD_df.corr('spearman'),'\n',CRISPOR_df.corr('spearman'),'\n',MIT_df.corr('spearman'))
log.info("Pearson value in K562 test set: CnnCrispr vs CFD vs CNN_std vs MIT")
log.info(df.corr('pearson'))
log.info(CFD_df.corr('pearson'))
log.info(CRISPOR_df.corr('pearson'))
log.info(MIT_df.corr('pearson'))
log.info("Spearman value in K562 test set: CnnCrispr vs CFD vs CNN_std vs MIT")
log.info(df.corr('spearman'))
log.info(CFD_df.corr('spearman'))
log.info(CRISPOR_df.corr('spearman'))
log.info(MIT_df.corr('spearman'))
#画ROC、PRC曲线
plt.figure(figsize=(15, 7))
roc_value,prc_value, ks = GetKS(K562_Y_test,K562_Y_score,label="CnnCrispr")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("CnnCrispr: ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(K562_Y_test,K562_CFD_value,label="CFD")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("CFD:ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
roc_value,prc_value, ks = GetKS(K562_Y_test,K562_CRISPOR_value,label="CNN_std")
print("ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
log.info("CNN_std:ROC_AUC=%0.3f,PRC_AUC=%0.3f,KS_value=%0.3f"%(roc_value,prc_value, ks))
plt.tight_layout()
plt.savefig(".../keras_model/Reg/%s/K562_set.png"%model_name)
plt.show()
plt.close()


model.save(".../keras_model/Reg/%s/model_%s.h5" %
           (model_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
model.save_weights('.../keras_model/Reg/%s/model_weights_%s.h5'% (model_name, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

log.info("model saved: %s_%s" % (model_name, datetime.datetime.now()))




'''
Pearson_value = pearson_r(Ytest, Ypred)
Spearman_value = spearman_co(Ytest,Ypred)
CFD_Pearson_value = pearson_r(Ytest, CFD_value)
CFD_Spearman_value = spearman_co(Ytest,CFD_value)
CRISPOR_Pearson_value = pearson_r(Ytest, CRISPOR_value)
CRISPOR_Spearman_value = spearman_co(Ytest,CRISPOR_value)
print("Pearson Value in all: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Pearson_value,CFD_Pearson_value,CRISPOR_Pearson_value))
print("Spearman Value in all: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Spearman_value,CFD_Spearman_value,CRISPOR_Spearman_value))
log.info("Pearson Value in all: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Pearson_value,CFD_Pearson_value,CRISPOR_Pearson_value))
log.info("Spearman Value in all: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Spearman_value,CFD_Spearman_value,CRISPOR_Spearman_value))
#Ypred = [int(i[1] > i[0]) for i in Ypred]
'''
'''
Pearson_value = pearson_r(hek_Ytest, hek_Ypred)
Spearman_value = spearman_co(hek_Ytest,hek_Ypred)
CFD_Pearson_value = pearson_r(hek_Ytest, hek_CFD_value)
CFD_Spearman_value = spearman_co(hek_Ytest,hek_CFD_value)
CRISPOR_Pearson_value = pearson_r(hek_Ytest, hek_CRISPOR_value)
CRISPOR_Spearman_value = spearman_co(hek_Ytest,hek_CRISPOR_value)
print("Pearson Value in Hek set: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Pearson_value,CFD_Pearson_value,CRISPOR_Pearson_value))
print("Spearman Value in Hek set: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Spearman_value,CFD_Spearman_value,CRISPOR_Spearman_value))
log.info("Pearson Value in Hek set: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Pearson_value,CFD_Pearson_value,CRISPOR_Pearson_value))
log.info("Spearman Value in Hek set: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Spearman_value,CFD_Spearman_value,CRISPOR_Spearman_value))
#hek_Ypred = [int(i[1] > i[0]) for i in hek_Ypred]
'''
'''
Pearson_value = pearson_r(K562_Ytest, K562_Ypred)
Spearman_value = spearman_co(K562_Ytest,K562_Ypred)
CFD_Pearson_value = pearson_r(K562_Ytest, K562_CFD_value)
CFD_Spearman_value = spearman_co(K562_Ytest,K562_CFD_value)
CRISPOR_Pearson_value = pearson_r(K562_Ytest, K562_CRISPOR_value)
CRISPOR_Spearman_value = spearman_co(K562_Ytest,K562_CRISPOR_value)
print("Pearson Value in K562 set: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Pearson_value,CFD_Pearson_value,CRISPOR_Pearson_value))
print("Spearman Value in K562 set: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Spearman_value,CFD_Spearman_value,CRISPOR_Spearman_value))
log.info("Pearson Value in K562 set: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Pearson_value,CFD_Pearson_value,CRISPOR_Pearson_value))
log.info("Spearman Value in K562 set: CnnCrispr_value=%0.3f, CFD_value=%0.3f, CRISPOR_value=%0.3f"%(Spearman_value,CFD_Spearman_value,CRISPOR_Spearman_value))
#K562_Ypred = [int(i[1] > i[0]) for i in K562_Ypred]\
'''