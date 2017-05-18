import tensorflow as tf
import numpy as np
import time
import datetime
import read_dataset_2d_cube
import matplotlib.pyplot as plt
import os.path

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

height, width, depth= 30, 30, 30

dataset_path="/home/penalvea/cubes/geometrics"

write_objects1="/home/penalvea/cubes/first"
ready_objects1="/home/penalvea/cubes/first_ready"
write_objects2="/home/penalvea/cubes/second"
ready_objects2="/home/penalvea/cubes/second_ready"
write_objects3="/home/penalvea/cubes/third"
ready_objects3="/home/penalvea/cubes/third_ready"
write_objects4="/home/penalvea/cubes/fourth"
ready_objects4="/home/penalvea/cubes/fourth_ready"
write_objects5="/home/penalvea/cubes/fifth"
ready_objects5="/home/penalvea/cubes/fifth_ready"
write_objects6="/home/penalvea/cubes/sixth"
ready_objects6="/home/penalvea/cubes/sixth_ready"
write_objects7="/home/penalvea/cubes/seventh"
ready_objects7="/home/penalvea/cubes/seventh_ready"
write_objects8="/home/penalvea/cubes/eighth"
ready_objects8="/home/penalvea/cubes/eighth_ready"
write_objects9="/home/penalvea/cubes/nineth"
ready_objects9="/home/penalvea/cubes/nineth_ready"
write_objects10="/home/penalvea/cubes/tenth"
ready_objects10="/home/penalvea/cubes/tenth_ready"


folder1="/first"
folder2="/second"
folder3="/third"
folder4="/fourth"
folder5="/fifth"
folder6="/sixth"
folder7="/seventh"
folder8="/eighth"
folder9="/nineth"
folder10="/tenth"



iterations_next_folder=50

iterations_complete=99999
run_until=datetime.datetime(2018, 10, 29, 13, 30)

output_path="/home/penalvea/NetResults/cubes/"

batch_size=8.0



def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def convPoolLayer_Red(inp, size, layersIn, layersOut,sizeRed, name):
    with tf.name_scope(name) as scope:
        w_conv=weight_variable([size, size, layersIn, layersOut])
        b_conv=bias_variable([layersOut])

        conv=tf.nn.relu(tf.nn.conv2d(inp, w_conv, strides=[1,1,1,1], padding="SAME")+b_conv)
        tf.summary.histogram(name + "/Filters", w_conv)

        pool=tf.nn.max_pool(conv, ksize=[1, sizeRed, sizeRed, 1], strides=[1, sizeRed, sizeRed, 1], padding="SAME")

    return pool


def convPoolLayer(inp, size, layersIn, layersOut, name):
    with tf.name_scope(name) as scope:
        w_conv = weight_variable([size, size, layersIn, layersOut])
        b_conv = bias_variable([layersOut])
        tf.summary.histogram(name + "/Filters", w_conv)
        conv = tf.nn.relu(tf.nn.conv2d(inp, w_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv)
    return conv


def denselyConnLayer(inp, layersIn, layersOut, name):
    with tf.name_scope(name) as scope:
        w_conv = weight_variable([layersIn, layersOut])
        b_conv = bias_variable([layersOut])
        tf.summary.histogram(name + "/Filters", w_conv)
        dense=tf.nn.relu(tf.matmul(inp, w_conv)+b_conv)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    return dropout

def denselyConnLayer_sigmoid(inp, layersIn, layersOut, name):
    with tf.name_scope(name) as scope:
        w_conv = weight_variable([layersIn, layersOut])
        b_conv = bias_variable([layersOut])
        tf.summary.histogram(name + "/Filters", w_conv)
        dense=tf.nn.sigmoid(tf.matmul(inp, w_conv)+b_conv)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    return dropout


def denselyConnLayerLineal(inp, layersIn, layersOut, name):
    with tf.name_scope(name) as scope:
        w_conv = weight_variable([layersIn, layersOut])
        b_conv = bias_variable([layersOut])
        tf.summary.histogram(name + "/Filters", w_conv)

        dense = tf.matmul(inp, w_conv) + b_conv
    return dense


def inference_4layers(inp):
    #conv1=convPoolLayer_Red(inp, 3, 1, 4, 3, 'conv1')
    #print conv1
    #conv2=convPoolLayer_Red(conv1, 3, 4, 8, 3, 'conv2')
    #print conv2
    #conv3 = convPoolLayer(conv2, 3, 8, 16)
    #print conv3
    #conv4 = convPoolLayer(conv3, 3, 80, 160)
    #print conv4


    flat=tf.reshape(inp, [-1,30*30*1])
    dens1=denselyConnLayer_sigmoid(flat, 30*30*1, 500, 'dense1_cube')
    dens2 = denselyConnLayer_sigmoid(dens1, 500, 50, 'dense2_cube')
    #dens3 = denselyConnLayer_sigomid(dens2, 1000, 100, 'dense3_cube')
    lineal1=denselyConnLayerLineal(dens2, 50, 6, 'lineal1_cube')

    return lineal1








sess=tf.InteractiveSession()
x=tf.placeholder("float", [None, height, width, 1])

x_input=tf.reshape(x, [-1, height, width, 1])
y_=tf.placeholder("float", shape=[None, 6])


res=inference_4layers(x_input)


with tf.name_scope('mean_squared_error'):

    loss_function = tf.reduce_sum(tf.losses.mean_squared_error(labels=y_, predictions=res))

#loss_function = -tf.reduce_sum(y_*tf.log(tf.nn.softmax(res) + 1e-10))

with tf.name_scope('train'):
    train_step = tf.contrib.layers.optimize_loss(loss_function, tf.contrib.framework.get_global_step(), optimizer='Adam', learning_rate=0.001)

epoch_loss=tf.Variable(1.0, name="epochLoss")
tf.summary.scalar("epochLoss", epoch_loss)

x_error=tf.Variable(0.0, name="epochXError")
tf.summary.scalar("epochXError", x_error)

y_error=tf.Variable(0.0, name="epochYError")
tf.summary.scalar("epochYError", y_error)

z_error=tf.Variable(0.0, name="epochZError")
tf.summary.scalar("epochZError", z_error)


rot_x_error=tf.Variable(0.0, name="epochRotXError")
tf.summary.scalar("epochRotXError", rot_x_error)

rot_y_error=tf.Variable(0.0, name="epochRotYError")
tf.summary.scalar("epochRotYError", rot_y_error)

rot_z_error=tf.Variable(0.0, name="epochRotZError")
tf.summary.scalar("epochRotZError", rot_z_error)







merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter("/home/penalvea/tensorboard/cubes/train", sess.graph)
test_writer=tf.summary.FileWriter("/home/penalvea/tensorboard/cubes/test")

saver = tf.train.Saver()

tf.global_variables_initializer().run()




### Train party hard

acum=0.0
count=0
i=0
epochs=0
start_time=init_time=time.time()
now=datetime.datetime.now()


x_acum=0.0
y_acum=0.0
z_acum=0.0
rot_x_acum=0.0
rot_y_acum=0.0
rot_z_acum=0.0


start_time = init_time = time.time()
now=datetime.datetime.now()
change=1
folder=1
objects=0
best=100.0
while(now<run_until):
    if change==1:
        if folder==1:
            while not os.path.isfile(ready_objects1):
              time.sleep(1)
            #os.remove(ready_objects1)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder1, height, width, depth)
            open(write_objects1, 'a').close()
            folder=2

        elif folder==2:
            while not os.path.isfile(ready_objects2):
              time.sleep(1)
            os.remove(ready_objects2)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder2, height, width, depth)
            open(write_objects2, 'a').close()
            folder=3

        elif folder==3:
            while not os.path.isfile(ready_objects3):
              time.sleep(1)
            os.remove(ready_objects3)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder3, height, width, depth)
            open(write_objects3, 'a').close()
            folder=4

        elif folder==4:
            while not os.path.isfile(ready_objects4):
              time.sleep(1)
            os.remove(ready_objects4)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder4, height, width, depth)
            open(write_objects4, 'a').close()
            folder=5

        elif folder==5:
            while not os.path.isfile(ready_objects5):
              time.sleep(1)
            os.remove(ready_objects5)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder5, height, width, depth)
            open(write_objects5, 'a').close()
            folder=6

        elif folder==6:
            while not os.path.isfile(ready_objects6):
              time.sleep(1)
            os.remove(ready_objects6)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder6, height, width, depth)
            open(write_objects6, 'a').close()
            folder=7

        elif folder==7:
            while not os.path.isfile(ready_objects7):
              time.sleep(1)
            os.remove(ready_objects7)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder7, height, width, depth)
            open(write_objects7, 'a').close()
            folder=8

        elif folder==8:
            while not os.path.isfile(ready_objects8):
              time.sleep(1)
            os.remove(ready_objects8)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder8, height, width, depth)
            open(write_objects8, 'a').close()
            folder=9

        elif folder==9:
            while not os.path.isfile(ready_objects9):
              time.sleep(1)
            os.remove(ready_objects9)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder9, height, width, depth)
            open(write_objects9, 'a').close()
            folder=10

        elif folder==10:
            while not os.path.isfile(ready_objects10):
              time.sleep(1)
            os.remove(ready_objects10)

            [training, validation] = read_dataset_2d_cube.readNextDataSet(dataset_path, folder10, height, width, depth)
            open(write_objects10, 'a').close()
            folder=1

        train_objects = training.num_examples
        val_objects = validation.num_examples
        change=0


    batch=training.next_batch(batch_size)


    objects+=len(batch[0])
    _ , output, label, result, summary=sess.run([train_step, loss_function, y_, res, merged], feed_dict={x: batch[0], y_: batch[1]})
    acum+=output
    #print(batch[1])
    #print (label, result)
    errors=np.sum(np.sqrt(np.power((label-result),2)),0)
    x_acum+=errors[0]
    y_acum+=errors[1]
    z_acum += errors[2]
    rot_x_acum+=errors[3]
    rot_y_acum+=errors[4]
    rot_z_acum += errors[5]




    if i>0 and i%int(train_objects/batch_size)==0:


        new_time = time.time()


	
        sess.run(epoch_loss.assign(acum/objects))
        sess.run(x_error.assign(x_acum/objects))
        sess.run(y_error.assign(y_acum / objects))
        sess.run(z_error.assign(z_acum / objects))
        sess.run(rot_x_error.assign(rot_x_acum / objects))
        sess.run(rot_y_error.assign(rot_y_acum / objects))
        sess.run(rot_z_error.assign(rot_z_acum / objects))

	
        epochs+=1
        train_writer.add_summary(summary, epochs)
	

        print ("iteration %d, loss_function: %.6f, elapsed time: %.2f, iteration time: %.2f" % (i/int(train_objects/batch_size) , acum/objects, (new_time-init_time)/60,  (new_time-start_time)/60))
        print ((x_acum/objects)*16, (y_acum/objects)*16, (z_acum/objects)*11, (rot_x_acum/objects)*3.1415, (rot_y_acum/objects)*3.1415, (rot_z_acum/objects)*3.1415)
        start_time = time.time()
        acum=0
        objects=0
        x_acum = 0.0
        y_acum = 0.0
        z_acum = 0.0
        rot_x_acum = 0.0
        rot_y_acum = 0.0
        rot_z_acum = 0.0
       
        if epochs%iterations_next_folder==0:
            change=0
            #saver.save(sess,output_path + "modelckp" + str(epochs) + ".ckpt")
	
        if epochs%10==0:
            acum_test=0
            objects_test=0
            x_acum_test = 0.0
            y_acum_test = 0.0
            z_acum_test = 0.0
            rot_x_acum_test = 0.0
            rot_y_acum_test = 0.0
            rot_z_acum_test = 0.0

            for j in range(int(val_objects/batch_size)):
                batch_test=validation.next_batch(batch_size)

                loss_test, label_test, result_test, summary = sess.run([loss_function, y_, res, merged], feed_dict={x: batch[0], y_: batch[1]})
                objects_test+=len(batch_test[0])
                acum_test+=loss_test
                errors = np.sum(np.sqrt(np.power((label - result), 2)), 0)
                x_acum_test += errors[0]
                y_acum_test += errors[1]
                z_acum_test += errors[2]
                rot_x_acum_test += errors[3]
                rot_y_acum_test += errors[4]
                rot_z_acum_test += errors[5]


            sess.run(epoch_loss.assign(acum_test/ objects_test))
            sess.run(x_error.assign(x_acum_test / objects_test))
            sess.run(y_error.assign(y_acum_test / objects_test))
            sess.run(z_error.assign(z_acum_test / objects_test))
            sess.run(rot_x_error.assign(rot_x_acum_test / objects_test))
            sess.run(rot_y_error.assign(rot_y_acum_test / objects_test))
            sess.run(rot_z_error.assign(rot_z_acum_test / objects_test))

            test_writer.add_summary(summary, epochs)
            print ("Test:    iteration %d, loss_function: %.6f" % (epochs, acum_test/objects_test))
            print ((x_acum_test / objects_test) * 16, (y_acum_test / objects_test) * 16, (z_acum_test / objects_test) * 11,(rot_x_acum_test / objects_test) * 3.1415, (rot_y_acum_test / objects_test) * 3.1415, (rot_z_acum_test / objects_test) * 3.1415)
            if best>(acum_test/objects_test):
                best=(acum_test/objects_test)
                saver.save(sess, output_path + "modelckp" + str(epochs) + ".ckpt")

    i = i + 1
    now = datetime.datetime.now()
