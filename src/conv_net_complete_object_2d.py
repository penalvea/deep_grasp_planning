#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import datetime
import read_dataset_2d
import matplotlib.pyplot as plt


height, width, depth= 30, 30, 30

dataset_path="/home/penalvea/dataset/geometrics"

folder="/first"

iterations_next_folder=500

iterations_complete=99999
run_until=datetime.datetime(2018, 10, 29, 13, 30)

output_path="/home/penalvea/NetResults/"

batch_size=10.0





keep_prob=tf.placeholder("float")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0.005, stddev=0.03)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.5, shape=shape)
    return tf.Variable(initial)

def convPoolLayer_Red(inp, size, layersIn, layersOut,sizeRed, name):
    with tf.name_scope(name):
        W_conv = weight_variable([size, size, layersIn, layersOut])
        b_conv = bias_variable([layersOut])

        #tf.summary.histogram(name + "/Filters",W_conv)

        preactivate=tf.nn.conv2d(inp, W_conv, strides= [1,1,1,1], padding='SAME')
        preactivate=tf.nn.bias_add(preactivate, b_conv)
        h_conv = tf.nn.crelu(preactivate, 'activation')
        h_pool = tf.nn.max_pool(h_conv,ksize=[1, sizeRed, sizeRed, 1], strides=[1, sizeRed, sizeRed, 1], padding='SAME')

    return h_pool


def convPoolLayer(inp, size, layersIn, layersOut, name):
        with tf.name_scope(name):
            W_conv = weight_variable([size, size, size, layersIn, layersOut])
            b_conv = bias_variable([layersOut])

            #tf.summary.histogram(name + "/Filters",W_conv)


            preactivate=tf.nn.conv2d(inp, W_conv, strides= [1,1,1,1], padding='SAME')
            preactivate=tf.nn.bias_add(preactivate, b_conv)
            h_conv = tf.nn.relu(preactivate, 'activation')
            h_pool = tf.nn.max_pool(h_conv,ksize=[1, size, size, 1],strides=[1, 1, 1, 1], padding='SAME')

        return h_pool


def denselyConnLayer(inp, layersIn, layersOut, name):
    with tf.name_scope(name):
        W_fc=weight_variable([layersIn, layersOut])
        b_fc=bias_variable([layersOut])
        h_dens_flat=tf.reshape(inp, [-1, layersIn])

        preactivate=tf.matmul(h_dens_flat, W_fc)
        preactivate=tf.nn.bias_add(preactivate,b_fc)
        h_fc=tf.nn.sigmoid(preactivate)
    return h_fc


def denselyConnLayerLineal(inp, layersIn, layersOut, name):
    with tf.name_scope(name):
        W_fc=weight_variable([layersIn, layersOut])
        b_fc=bias_variable([layersOut])
        h_dens_flat=tf.reshape(inp, [-1, layersIn])

        h_fc=tf.matmul(h_dens_flat, W_fc)
        h_fc=tf.nn.bias_add(h_fc,b_fc)

    return h_fc


def inference_4layers(inp):
    #print("using 4 layers inference")
    conv1 = convPoolLayer_Red (inp, 5, 1, 100, 5, 'conv1')
    #print conv1
    #conv2 = convPoolLayer_Red (conv1, 3, 30, 50, 3, 'conv2')
    #print conv2
    #conv3 = convPoolLayer_Red (conv2, 3, 100, 150, 3,'conv3')
    #conv4 = convPoolLayer (conv3, 3, 32, 64, 'conv4')

    print conv1


    #dens1 = denselyConnLayer(inp, height*width*depth, 2000, 'dens1')
    #dens2 = denselyConnLayerLineal(conv1, height*width*depth, 'dens2')


    res=tf.reshape(conv1, [-1, height, width, depth, 1])

    return res


def inference_4layersNoRed(inp):
    print("using 4 layers inference")
    conv1 = convPoolLayer(inp, 5, 1, 20, 'conv1')
    conv2 = convPoolLayer(conv1, 5, 20, 1, 'conv2')
    #conv3 = convPoolLayer(conv2, 5, 24, 32,'conv3')
    #conv4 = convPoolLayer (conv3, 5, 32, 1, 'conv4')

    return conv2



[training, validation]=read_dataset_2d.readNextDataSet(dataset_path, folder, height, width, depth)
train_objects=training.num_examples
val_objects=validation.num_examples


sess=tf.InteractiveSession()
x=tf.placeholder("float", [None, height, width, depth, 1])

x_input=tf.reshape(x, [-1, height, width, depth, 1])
y_=tf.placeholder("float", shape=[None, height, width, depth, 1])


res=inference_4layers(x_input)


#ones=tf.ones([None, height, width, depth, 1], tf.float32)



# Y Final
with tf.name_scope("finalY") as scope:
    y_result=tf.reshape(res,[-1, height,width,depth,1])
    #y_result=res

#Loss Function
#with tf.name_scope("loss_function"):

    #diff=tf.nn.sigmoid_cross_entropy_with_logits(None, y_result, y_)

    #log=tf.log(y_result)
    #print log

    #positive= tf.multiply(tf.negative(y_),tf.log(y_result))
    #negative=tf.subtract(ones, y_)
    #negative= tf.multiply(tf.subtract(ones, y_), tf.log(tf.subtract(ones, y_result)))
    #diff=tf.negative(tf.subtract(positive, negative))
    #diff=tf.negative(positive)
    #y_cast=tf.reshape(tf.cast(y_, tf.int32), [ int (batch_size*height*width*depth)])
    #y_res_reshape=tf.reshape(y_result, [int (batch_size*height*width*depth)])
    #loss_function=tf.losses.sparse_softmax_cross_entropy(y_cast, y_res_reshape)
    #loss_function=tf.reduce_sum(diff)

    #loss_function=-tf.reduce_sum(y_*tf.log(y_result + 1e-10))
    #loss_function=tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_, y_result))




    #loss_function=tf.reduce_mean(tf.abs(tf.subtract(y_result, y_))*10)
    #y_result=tf.abs(y_result)
    y_round = tf.round(y_result)
    #loss_function=(tf.reduce_sum(y_*(1-y_result))*10)+ tf.reduce_sum((1-y_)*y_result)




    y_change = tf.reshape(y_, [-1, height*width*depth])
    y_result_change=tf.reshape(y_result, [-1, height*width*depth])

    y_target= tf.reshape(y_,[-1, height * width * depth])
    #loss_function=((tf.reduce_sum(tf.abs(y_change*tf.abs((1-y_result_change))))))+tf.reduce_sum(tf.abs((1-y_change)*tf.abs(y_result_change)))
    loss_function=tf.reduce_mean(tf.square(res - y_target))


    #loss_function=tf.reduce_sum(tf.losses.absolute_difference(y_change, y_result_change))
    bad_results=tf.reduce_sum(tf.abs(tf.subtract(y_round, y_)))


with tf.name_scope("train"):
    train_step=tf.train.AdamOptimizer(1e-4).minimize(loss_function)

#lossbatch=tf.Variable(3.0, name="lossbatch")
#tf.summary.scalar("loss batch", lossbatch)


saver = tf.train.Saver()
#merged = tf.summary.merge_all()
#writer = tf.summary.FileWriter("/tmp/geometric_logs", sess.graph)
tf.global_variables_initializer().run()




### Train party hard

acum=0
count=0
i=0
epochs=0
start_time=init_time=time.time()
now=datetime.datetime.now()
bad=0
unos = 0
while(now<run_until):
    batch=training.next_batch(batch_size)






    _ , output, label, result, malos, rounded=sess.run([train_step, loss_function, y_, y_result, bad_results, y_round], feed_dict={x: batch[1], y_: batch[1]})


    #for m in range(len(result)):


        #unos_real=0
       # malos=0
        #for j in range(50):
        #    for k in range(50):
        #        for l in range(50):
        #            if result[m][j][k][l][0]>0.5:
        #                unos+=1
        #            if batch[1][m][j][k][l][0]==1:
        #                unos_real+=1
                 #  if result[m][j][k][l][0]>0.5 and label[m][j][k][l][0]==1:
                 #       buenos+=1
                 #   if result[m][j][k][l][0] < 0.5 and label[m][j][k][l][0] == 1:
                 #       malos+=1
                 #   if result[m][j][k][l][0] > 0.5 and label[m][j][k][l][0] == 0:
                 #       malos+=1
                 #   if result[m][j][k][l][0] < 0.5 and label[m][j][k][l][0] == 0:
                 #       buenos+=1
        #print (buenos, malos)
       # print (unos, unos_real)
       # unos=0
       # unos_real=0
    bad+=malos
    acum+=output
    if(epochs<training.epochs_completed):
        aux1 = open("/home/penalvea/au1.txt", 'w')
        aux2 = open("/home/penalvea/au2.txt", 'w')
        aux3 = open("/home/penalvea/au3.txt", 'w')
        for a in range(height):
            for b in range(width):
                for c in range(depth):
                    if rounded[0][a][b][c][0]==1:
                        unos+=1
                    #if int(batch[1][0][a][b][c][0])==1:
                        #print (int(batch[1][0][a][b][c][0]), rounded[0][a][b][c][0], result[0][a][b][c][0])
                    #value = batch[0][0][a][b][c][0]
                    #aux1.write(str(int(batch[0][0][a][b][c][0])) + " ")
                    aux2.write(str(int(batch[1][0][a][b][c][0])) + " ")
                    aux3.write(str(int(rounded[0][a][b][c][0])) + " ")
                    #print result[0][a][b][c][0]
        aux1.close()
        aux2.close()
        aux3.close()
        print (unos)
        print "salgo"
        new_time=time.time()
        print("epoch %d error: %.6f elapsed time: %.2f iteration time %.2f" % (epochs , acum/train_objects , (new_time-init_time)/60 ,(new_time - start_time)/60 ) )
        print (bad/train_objects)
        start_time=time.time()
        epochs=training.epochs_completed
        acum=0
        count=count+1
        bad=0
        unos=0

        for i in range(4):
            for j in range(i+1, 4):

                print (len(rounded), len(result))
                aaa=abs(rounded[i] - rounded[j])
                print (i, j, aaa.sum())
