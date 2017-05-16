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
    def __init__(self, dataset_path,  objects, labels, height, width, depth):
        self.dataset_path_=dataset_path
        self.objects_=objects
        self.height_=int (height)
        self.width_=int (width)
        self.depth_=int (depth)
        self.labels_=labels

        self.num_examples_ = len(objects)
        self.data_=self.read_data()



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
            shuffle(self.data_)

        end=int(self.num_examples_)
        self.index_in_epoch_+=batch_size

        if self.index_in_epoch_>=self.num_examples:
            #Finished epoch
            self.epochs_completed_+=1
            self.index_in_epoch_=0
        else:
            end=int(self.index_in_epoch_)
        return self.get_input_and_label(start, end)

    def get_input_and_label(self, start, end):
        batch=self.data_[int(start):int(end)][:]
        inputs=np.zeros((int(end-start), self.height_, self.width_, 1))
        labels=np.zeros((int(end-start),4), "float")
        for i in range(0,len(batch)):
            inputs[i]=batch[int(i)][0]

            labels[i]=batch[int(i)][1]

        return inputs, labels



    def read_data(self):
        data=[]

        file_labels = open(self.labels_, "r")


        end=len(self.objects_)
        #for i in range(len(self.objects_)):
        for i in range(end):
            if i%1000==0:
                print(i)
            inputs = np.zeros([self.height_, self.width_, 1], "float")
            labels = np.zeros([4], "float")

            file_inputs = open(self.dataset_path_ + "/side_objects/" + self.objects_[i], "r")

            line_input = file_inputs.readline()

            values_input = line_input.split(" ")
	    


            if len(values_input)<(self.height_*self.width_*self.depth_)+1:
                print ("Error %d number of data" %  len(values_input))
            else:
                for j in range(self.height_):
                    for k in range(self.width_):
                        first = False
                        for l in range(self.depth_):
                            if not first:
                                if values_input[(j * self.width_ * self.depth_) + (k * self.depth_) + l] == "1":
                                    inputs[j][k][0] = float(l) / self.depth_
                                    first = True

                file_labels.seek(0)
                for line_labels in file_labels:
                    values_labels = line_labels.split(" ")
                    #print (self.objects_[i].split(".")[0])
                    if values_labels[0] == self.objects_[i].split(".")[0]:

                        if values_labels[1] == "cylinder":



                            labels[0] = (float(values_labels[3])-4)/16
                            labels[1] = (float(values_labels[2])-2)/6
                            labels[2] = float(values_labels[5])/3.1415
                            labels[3] = float(values_labels[6])/3.1415


                        else:
                            print ("ninguna")

                data.append([inputs, labels])
            self.num_examples_=len(data)
        return data


























































    def read_input_and_label(self, start, end):
        inputs=np.zeros([int(end-start), self.height_, self.width_, 1], "float")

        labels=np.zeros([int(end-start), 4], "float")


        file_labels=open(self.labels_, "r")


        for i in range(int(start), int(end)):
            file_inputs=open(self.dataset_path_+"/side_objects/"+self.objects_[i], "r")


            line_input=file_inputs.readline()

            values_input=line_input.split(" ")


            for j in range(self.height_):
                for k in range(self.width_):
                    first=False
                    for l in range(self.depth_):
                        if not first:
                            if values_input[(j*self.width_*self.depth_)+(k*self.depth_)+l]=="1":
                                inputs[int(i-start)][j][k][0]=float (l)/self.depth_
                                first=True

            file_labels.seek(0)
            for line_labels in file_labels:
                values_labels=line_labels.split(" ")
                if values_labels[0]==self.objects_[i].split(".")[0]:

                    if values_labels[1]=="cube":
                        labels[int (i-start)][0]=1.0
                    elif values_labels[1]=="cylinder":
                        labels[int(i - start)][1] = 1.0
                    elif values_labels[1] == "cone":
                        labels[int(i - start)][2] = 1.0
                    elif values_labels[1] == "sphere":
                        labels[int(i - start)][3] = 1.0


        return inputs, labels




def readNextDataSet(dataset_path, folder, height, width, depth):


    train_objects=listdir(dataset_path+folder+"/training/side_objects")
    validation_objects=listdir(dataset_path+folder+"/validation/side_objects")
    train_labels=dataset_path+folder+"/training/types.txt"
    validation_labels=dataset_path+folder+"/validation/types.txt"

    return [DataSet(dataset_path+folder+"/training", train_objects, train_labels, height, width, depth), DataSet(dataset_path+folder+"/validation", validation_objects, validation_labels, height, width, depth)]
