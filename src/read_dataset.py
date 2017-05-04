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
    def __init__(self, dataset_path,  objects, height, width, depth):
        self.dataset_path_=dataset_path
        self.objects_=objects
        self.height_=int (height)
        self.width_=int (width)
        self.depth_=int (depth)


        self.num_examples_=len(objects)

        self.epochs_completed_=0
        self.index_in_epoch_=0

    @property
    def num_examples(self):
        return self.num_examples_

    @property
    def epochs_completed(self):
        return self.epochs_completed_

    @property
    def index(self):
        return self.index_in_epoch_

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start=self.index_in_epoch_
        if start==0:
            shuffle(self.objects_)

        end=int(self.num_examples_-1)
        self.index_in_epoch_+=batch_size

        if self.index_in_epoch_>=self.num_examples:
            #Finished epoch
            self.epochs_completed_+=1
            self.index_in_epoch_=0
        else:
            end=int(self.index_in_epoch_)
        return self.read_input_and_label(start, end)

    def read_input_and_label(self, start, end):
        inputs=np.zeros([int(end-start), self.height_, self.width_, self.depth_, 1], "float")
        labels=np.zeros([int(end-start), self.height_, self.width_, self.depth_, 1], "float")

        for i in range(int(start), int(end)):
            file_inputs=open(self.dataset_path_+"/side_objects/"+self.objects_[i], "r")
            file_labels=open(self.dataset_path_+"/complete_objects/"+self.objects_[i], "r")
            #print (self.dataset_path_+"/side_objects/"+self.objects_[i])
            #print (self.dataset_path_+"/complete_objects/"+self.objects_[i])

            line_input=file_inputs.readline()
            line_label=file_labels.readline()

            values_input=line_input.split(" ")
            values_label=line_label.split(" ")
            #print (len(values_input), len(values_label))

            for j in range(self.height_):
                for k in range(self.width_):
                    for l in range(self.depth_):
                        inputs[int(i-start)][j][k][l][0]=float (values_input[(j*self.width_*self.depth_)+(k*self.depth_)+l])

                        labels[int (i-start)][j][k][l][0]=float (values_label[(j*self.width_*self.depth_)+(k*self.depth_)+l])
                        #print(labels[int(i - start)][j][k][l][0])
        return inputs, labels




def readNextDataSet(dataset_path, folder, height, width, depth):

    train_list=[]
    val_list=[]
    num_object=0
    train_objects=listdir(dataset_path+folder+"/training/side_objects")
    validation_objects=listdir(dataset_path+folder+"/validation/side_objects")

    return [DataSet(dataset_path+folder+"/training", train_objects, height, width, depth), DataSet(dataset_path+folder+"/validation", validation_objects, height, width, depth)]




