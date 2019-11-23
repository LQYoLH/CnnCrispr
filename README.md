# CnnCrispr
A deep learning method for sgRNA off-target propensity prediction.

## Introduction
CnnCrispr is a deep learning method for sgRNA off-target propensity prediction. It automatically trained the sequence features of sgRNA-DNA pairs with GloVe model, and embeded the trained word vector matrix into the deep learning model including biLSTM and CNN with five hidden layers. 
## Requirement
*python == 3.6
*tensorflow == 1.13.1
*keras == 2.2.4

## Usage
1. Encode the sgrna-dna pairs  using the method mentioned in our paper.
2. Load .H5 model and make prediction.
