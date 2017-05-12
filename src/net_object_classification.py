import tensorflow as tf
import numpy as np
import time
import datetime
import read_dataset_2d_object
import matplotlib.pyplot as plt
import os.path

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

height, width, depth= 60, 60, 60

dataset_path="/home/penalvea/dataset2/geometrics"

write_objects1="/home/penalvea/dataset2/first"
ready_objects1="/home/penalvea/dataset2/first_ready"
write_objects2="/home/penalvea/dataset2/second"
ready_objects2="/home/penalvea/dataset2/second_ready"


folder1="/first"
folder2="/second"

iterations_next_folder=500

iterations_complete=99999
run_until=datetime.datetime(2018, 10, 29, 13, 30)

output_path="/home/penalvea/NetResults/"

batch_size=20.0



def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def convPoolLayer_Red(inp, size, layersIn, layersOut,sizeRed):
    w_conv=weight_variable([size, size, layersIn, layersOut])
    b_conv=bias_variable([layersOut])

    conv=tf.nn.relu(tf.nn.conv2d(inp, w_conv, strides=[1,1,1,1], padding="SAME")+b_conv)

    pool=tf.nn.max_pool(conv, ksize=[1, sizeRed, sizeRed, 1], strides=[1, sizeRed, sizeRed, 1], padding="SAME")

    return pool


def convPoolLayer(inp, size, layersIn, layersOut):
    w_conv = weight_variable([size, size, layersIn, layersOut])
    b_conv = bias_variable([layersOut])

    conv = tf.nn.relu(tf.nn.conv2d(inp, w_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv)
    return conv


def denselyConnLayer(inp, layersIn, layersOut):
    w_conv = weight_variable([layersIn, layersOut])
    b_conv = bias_variable([layersOut])

    dense=tf.nn.relu(tf.matmul(inp, w_conv)+b_conv)

    return dense


def denselyConnLayerLineal(inp, layersIn, layersOut):
    w_conv = weight_variable([layersIn, layersOut])
    b_conv = bias_variable([layersOut])

    dense = tf.matmul(inp, w_conv) + b_conv
    return dense


def inference_4layers(inp):
    conv1=convPoolLayer_Red(inp, 5, 1, 20, 3)
    print conv1
    conv2=convPoolLayer_Red(conv1, 3, 20, 40, 3)
    print conv2
    conv3 = convPoolLayer(conv2, 3, 40, 80)
    print conv3
    #conv4 = convPoolLayer(conv3, 3, 80, 160)
    #print conv4


    flat=tf.reshape(conv3, [-1,7*7*80])
    dens1=denselyConnLayer(flat, 7*7*80, 2048)
    dens2=denselyConnLayerLineal(dens1, 2048, 4)

    return dens2








sess=tf.InteractiveSession()
x=tf.placeholder("float", [None, height, width, 1])

x_input=tf.reshape(x, [-1, height, width, 1])
y_=tf.placeholder("float", shape=[None, 4])


res=inference_4layers(x_input)

print (res)
print (y_)

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=res))
#loss_function = -tf.reduce_sum(y_*tf.log(tf.nn.softmax(res) + 1e-10))

train_step = tf.contrib.layers.optimize_loss(loss_function, tf.contrib.framework.get_global_step(), optimizer='Adam', learning_rate=0.01)



# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)

#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)
correct_prediction = tf.equal(tf.argmax(res, 1), tf.argmax(y_, 1))






saver = tf.train.Saver()

tf.global_variables_initializer().run()




### Train party hard

acum=0
count=0
i=0
epochs=0
start_time=init_time=time.time()
now=datetime.datetime.now()


predictions=[]


start_time = init_time = time.time()
now=datetime.datetime.now()
change=1
while(now<run_until):
    if change==1:
        if epochs%100==0:
            while not os.path.isfile(ready_objects1):
              time.sleep(1)
            os.remove(ready_objects1)
            open(write_objects2, 'a').close()
            [training, validation] = read_dataset_2d_object.readNextDataSet(dataset_path, folder1, height, width, depth)
        else:
            while not os.path.isfile(ready_objects2):
              time.sleep(1)
            os.remove(ready_objects2)
            open(write_objects1, 'a').close()
            [training, validation] = read_dataset_2d_object.readNextDataSet(dataset_path, folder2, height, width, depth)

        train_objects = training.num_examples
        val_objects = validation.num_examples
        change=0


    batch=training.next_batch(batch_size)

    #print (batch[1])

    _ , output, label, result, correct_pred=sess.run([train_step, loss_function, y_, res, correct_prediction,], feed_dict={x: batch[0], y_: batch[1]})
    acum+=output
    predictions.extend(correct_pred)
    #print (label)

    if i>0 and i%int(train_objects/batch_size)==0:
        new_time = time.time()

        accuracy=np.mean(correct_pred,  dtype=np.float64)

        print ("iteration %d, loss_function: %.6f, correct_predictions: %f, elapsed time: %.2f, iteration time: %.2f" % (i/int(train_objects/batch_size) , acum,  accuracy, (new_time-init_time)/60,  (new_time-start_time)/60))
        start_time = time.time()
        acum=0
        epochs+=1
        if epochs%50==0:
            change=1

    i = i + 1
    now = datetime.datetime.now()
