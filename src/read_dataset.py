# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
from os import listdir
import math
from collections import namedtuple
from Crypto.Random.random import shuffle




class DataSet(object):
    def __init__(self, data_set_path, height, width, depth):
        self.data_set_path=data_set_path



def readNextDataSet(dataset_path, folder, height, width, depth):

    train_list=[]
    val_list=[]
    num_object=0
    train_objects=listdir(dataset_path+folder+"/train/input")


