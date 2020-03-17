# CnnCrispr
A deep learning method for sgRNA off-target propensity prediction.

## Introduction
CnnCrispr is a deep learning method for sgRNA off-target propensity prediction. 

It automatically trained the sequence features of sgRNA-DNA pairs with GloVe model, and embeded the trained word vector matrix into the deep learning model including biLSTM and CNN with five hidden layers. 

## Requirement
* python == 3.6
* tensorflow == 1.13.1
* keras == 2.2.4

## Usage
1. Encode the sgrna-dna pairs  using the method mentioned in our paper.
2. Load .H5 model and make prediction.
3. All the code and examples you need for prediction can be seen in CnnCrispr_final.

## File description

* Comparison models store three existing models for model Comparison: CFD, MIT, CNN_std.
* CFD_get.py: source code downloaded from the literature and slightly modified according to the requirements.(GitHub link)
* MITsourcecode.Py: sourcecode downloaded from the literature with minor modifications to the requirements;(GitHub link)
* CnnCrispr_code contains all the codes including CnnCrispr training and model comparison. Readers can choose the codes they need.There are 8 python files in this folder:
* data_proprecess.py: preprocessed the original data into a vector form and trained the GloVe model.
* data_preprocess_leave_one_out.py: sequence preprocessing of the original data, transformation to vector form, and GloVe model training.
* loaddata.py: GloVe model embedding and data set partition command.
* Model_get.py: stores the specific structure of deep learning model.
* Model_eval.py: used to evaluate model performance.
* Model_twoset.py: model training and testing in a ratio of 8 to 2 (classification schema).
* Model_twoset_reg.py: model training and testing in a ratio of 8 to 2 (regression schema).
* Model_class_leave_one_out.py: Leave_one_sgRNA_out cross validation for model comparison.
