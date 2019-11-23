import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import auc

#-----------------------------------------------------
#def model sensitivity
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

#def model specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def pr_auc(y_true, y_pred):
    recall = tf.stack([binary_recall(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    precision = tf.stack([binary_precision(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    precision = tf.concat([tf.ones((1,)), precision], axis=0)
    binSizes = -(precision[1:] - precision[:-1])
    s = recall * binSizes
    return K.sum(s, axis=0)

def roc_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')#cast 数据类型转换为浮点型
    N = K.sum(1 - y_true)#计算真实样本中零值的个数
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier


def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    return TP / P


def binary_recall(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')#cast 数据类型转换为浮点型
    P = K.sum(y_true)#计算真实样本中零值的个数
    TP = K.sum(y_pred * y_true)
    return TP / P

def binary_precision(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')#cast 数据类型转换为浮点型
    #P = K.sum(y_true)#计算真实样本中零值的个数
    FP = K.sum(y_pred - y_pred * y_true)
    TP = K.sum(y_pred * y_true)
    return TP / (TP+FP)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# result report of all



def model_rep(y_true, y_pred):
    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1score = f1_score(y_true, y_pred, average='binary')
    return p, r, f1score


def model_matrix(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)


def roc_Curve(y_pred,y_true,title="The test result",label="CnnCrispr"):
    FPR = []
    TPR = []
    Precision = []
    Recall = []
    # print(len(y_true))
    for threshold in [i/100 for i in range(0,100,1)]:
        threshold = np.float(threshold)
        TP=0
        TN=0
        FP=0
        FN = 0
        for i in range(len(y_true)):
            # print(y_true[i],threshold)
            if y_true[i] == 1:
                if y_pred[i]<threshold:
                    FN = FN +1
                else:
                    TP = TP +1
            else:
                if y_pred[i]<threshold:
                    TN = TN +1
                else:
                    FP = FP +1
        #print(TP, TN, FP, FN)
        if TP+FN ==0:
            continue
        elif FP+TN==0:
            continue
        else:
            FPR.append(FP / (FP + TN))
            TPR.append(TP / (TP + FN))
        if TP+FP ==0:
            continue
        elif TP+FN ==0:
            continue
        else:
            Precision.append(TP / (TP + FP))
            Recall.append(TP / (TP + FN))
    c = list(zip(FPR, TPR))
    c.sort(reverse=False)
    FPR[:], TPR[:] = zip(*c)
    value = auc(FPR, TPR)
    d = list(zip(Recall, Precision))
    # d.sort(reverse=True)
    d.sort(key=lambda x: (x[0], -x[1]))  # sorted(a, key=lambda x: (x[0], -x[1]))
    # print(d)
    Recall[:], Precision[:] = zip(*d)
    PR_auc = auc(Recall, Precision)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(FPR, TPR,'b-',label='(%s_AUC=%0.3f)'%(label,value))
    plt.legend()
    plt.title(title)
    plt.show()

    plt.plot([0, 1], [1, 0], 'k--')
    plt.plot(Recall,Precision,'r-',label="(%s_AUC=%0.3f)"%(label,PR_auc))
    plt.legend()
    plt.title(title)
    plt.show()
    return value,PR_auc


#--------------------------------------------------------------

def ComuTF(y_pred, y_true):
    TP = sum([1 if a==b==1 else 0 for a, b in zip(y_pred,y_true)])
    FP = sum([1 if a==1 and b==0 else 0 for a,b in zip(y_pred,y_true)])
    TN = sum([1 if a==b==0 else 0 for a, b in zip(y_pred,y_true)])
    FN = sum([1 if a==0 and b==1 else 0 for a, b in zip(y_pred,y_true)])

    TPR = TP/(TP+FN)
    FPR = FP/(TN+FP)
    return TPR-FPR

def GetKS(y_true, y_pred_prob,label="CnnCrispr"):
    fpr, tpr, thresholds =roc_curve(y_true,y_pred_prob)
    roc_value = auc(fpr,tpr)
    ks = max(tpr-fpr)
    pre, recall, thresholds1 = precision_recall_curve(y_true,y_pred_prob)
    prc_value = auc(recall,pre)
    #print(roc_value)


    plt.subplot(121)
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr,tpr,label="%s(AUC=%0.3f)"%(label,roc_value))
    plt.legend(loc=4,fontsize=14)
    plt.xlabel("FPR",fontsize=14)
    plt.ylabel("TPR",fontsize=14)
    plt.title("The ROC Curve",fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(122)
    plt.plot([0, 1], [1, 0], 'k--')
    plt.plot(recall, pre, label="%s(AUC=%0.3f)" % (label,prc_value))
    plt.xlabel("Precision",fontsize=14)
    plt.ylabel("Recall",fontsize=14)
    plt.title("The PRC Curve",fontsize=20)
    plt.legend(loc=1,fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    '''
    plt.figure(3)
    plt.plot(tpr,label="tpr")
    plt.plot(fpr,label="fpr")
    plt.plot(tpr-fpr,label="KS")
    plt.title("The KS Curve")
    plt.legend()
    #plt.show()
    '''
    return roc_value,prc_value, ks


