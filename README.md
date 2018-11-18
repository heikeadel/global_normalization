# Description

This code was written in 2017 for the experiments presented in the
paper "Global Normalization of Convolutional Neural Networks for 
Joint Entity and Relation Classification"
which was published at EMNLP 2017.

The author of the code is Heike Adel. Some parts of the code are 
based on the theano tutorials (http://deeplearning.net/tutorial/)
and the CRF layer implementation is based on https://github.com/glample/tagger.

# Usage

For usage, please follow these steps:

1. create a fuel dataset:

python createDataStream_setupX.py config

Please refer to the paper for the different setups.
The config files used in the paper can be found in the folder configs.

2. train and evaluate the model:

python train.py config

Use the same config file as above. train.py is used for setup 1 and 2,
train_setup3.py is used for setup 3.

# Citation

If you use the code for your work, please cite the following paper:

```
@inproceedings{globalAdel2017,
  author = {Heike Adel and Hinrich Sch\"{u}tze},
  title = {Global Normalization of Convolutional Neural Networks for
Joint Entity and Relation Classification},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics}
}
```
