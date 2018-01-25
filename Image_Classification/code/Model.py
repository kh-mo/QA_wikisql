import os
import sys
import Block as b
import numpy as np
import tensorflow as tf
sys.path.append(os.getcwd() + "/Image_Classification/code/")

class Basic(b.Block):
    def __init__(self):
        ## 네트워크 구조 지정
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.bol = tf.placeholder(tf.bool)

        self.conv1 = self.Conv(input=self.x, filter=[11,11,3], output_channel=96, stride=4, init="xavier", trainable=True, name="conv1")
        self.batch1 = self.BN(input=self.conv1, trainable=self.bol, name="batch1")
        self.act1 = self.Act(input=self.batch1, function="relu", name="relu1")
        self.pool1 = self.Pool(input=self.act1, filter=2, stride=2, function="max", name="pool1")

        self.conv2 = self.Conv(input=self.pool1, filter=[5,5,96], output_channel=256, stride=1, init="xavier", trainable=True, name="conv2")
        self.batch2 = self.BN(input=self.conv2, trainable=self.bol, name="batch2")
        self.act2 = self.Act(input=self.batch2, function="relu", name="relu2")
        self.pool2 = self.Pool(input=self.act2, filter=2, stride=2, function="max", name="pool2")

        self.conv3 = self.Conv(input=self.pool2, filter=[3,3,256], output_channel=384, stride=1, init="xavier", trainable=True, name="conv3")
        self.batch3 = self.BN(input=self.conv3, trainable=self.bol, name="batch3")
        self.act3 = self.Act(input=self.batch3, function="relu", name="relu3")

        self.conv4 = self.Conv(input=self.act3, filter=[3,3,384], output_channel=384, stride=1, init="xavier", trainable=True, name="conv4")
        self.batch4 = self.BN(input=self.conv4, trainable=self.bol, name="batch4")
        self.act4 = self.Act(input=self.batch4, function="relu", name="relu4")

        self.conv5 = self.Conv(input=self.act4, filter=[3,3,384], output_channel=384, stride=1, init="xavier", trainable=True, name="conv5")
        self.batch5 = self.BN(input=self.conv5, trainable=self.bol, name="batch5")
        self.act5 = self.Act(input=self.batch5, function="relu", name="relu5")
        self.pool3 = self.Pool(input=self.act5, filter=2, stride=2, function="max", name="pool5")

        self.fc1 = self.FC(input=self.pool3, output_channel=2048, init="xavier", trainable=True, name="fc1")
        self.batch6 = self.BN(input=self.fc1, trainable=self.bol, name="batch6")
        self.act6 = self.Act(input=self.batch6, function="relu", name="relu6")

        self.fc2 = self.FC(input=self.act6, output_channel=2048, init="xavier", trainable=True, name="fc2")
        self.batch7 = self.BN(input=self.fc2, trainable=self.bol, name="batch7")
        self.act7 = self.Act(input=self.batch7, function="relu", name="relu7")

        self.fc3 = self.FC(input=self.act7, output_channel=2, init="xavier", trainable=True, name="fc3")

        ## cost, accuracy, train 지정
        with tf.variable_scope("cost"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc3))
            tf.summary.scalar("cross_entropy", cross_entropy)

        with tf.variable_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fc3, 1), tf.argmax(self.y, 1)), dtype=tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        ema_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope("training"):
            with tf.control_dependencies(ema_op):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        self.probability = tf.nn.softmax(self.fc3)

class Alexnet():
    def __init__(self):
        print("아직 미완성입니다.")

class Vggnet():
    def __init__(self):
        print("아직 미완성입니다.")

class Googlexnet():
    def __init__(self):
        print("아직 미완성입니다.")

class Resnet():
    def __init__(self):
        print("아직 미완성입니다.")

class DenseNet(b.Block):
    def __init__(self, number_of_label, growth_rate):
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
        self.y = tf.placeholder(tf.float32, shape=[None, number_of_label], name="output")
        self.bol = tf.placeholder(tf.bool, name="bol")
        self.number_of_label = number_of_label
        self.growth_rate = growth_rate
        self.learning_rate = 1e-4

        self.conv1 = self.Conv(input=self.x, filter=[7,7,3], output_channel=2*self.growth_rate, stride=2, init="xavier", trainable=True, name="conv1")
        self.pool1 = self.Pool(input=self.conv1, filter=3, stride=2, function="max", name="pool1")

        self.dense1 = self.dense_block(input=self.pool1, number_of_block=6, name="dense1")
        self.trans1 = self.transition_layer(input=self.dense1, name="trans1")

        self.dense2 = self.dense_block(input=self.trans1, number_of_block=12, name="dense2")
        self.trans2 = self.transition_layer(input=self.dense2, name="trans2")

        self.dense3 = self.dense_block(input=self.trans2, number_of_block=24, name="dense3")
        self.trans3 = self.transition_layer(input=self.dense3, name="trans3")

        self.dense4 = self.dense_block(input=self.trans3, number_of_block=16, name="dense4")
        self.pool2 = self.Pool(input=self.dense4, filter=7, stride=7, function="avg", name="pool2")

        self.fc = self.FC(input=self.pool2, output_channel=self.number_of_label, init="xavier", trainable=True, name="fc")
        self.probability = tf.nn.softmax(self.fc)

        ## cost, accuracy, train 지정
        with tf.variable_scope("cost"):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc))
            tf.summary.scalar("cross_entropy", cross_entropy)

        with tf.variable_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fc, 1), tf.argmax(self.y, 1)), dtype=tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        ema_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope("training"):
            with tf.control_dependencies(ema_op):
                train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

    def dense_block(self, input, number_of_block, name):
        with tf.variable_scope(name):
            for iter in range(number_of_block):
                bn1 = self.BN(input=input, trainable=self.bol, name=name+"_bn1_"+str(iter+1))
                act1 = self.Act(input=bn1, function="relu", name=name+"_relu1_"+str(iter+1))
                conv1 = self.Conv(input=act1, filter=[1, 1, act1.get_shape().as_list()[3]],
                                  output_channel=4*self.growth_rate, stride=1, init="xavier",
                                  trainable=True, name=name+"_conv1_"+str(iter+1))
                bn2 = self.BN(input=conv1, trainable=self.bol, name=name+"_bn2_"+str(iter+1))
                act2 = self.Act(input=bn2, function="relu", name=name+"_relu2_"+str(iter+1))
                conv2 = self.Conv(input=act2, filter=[3, 3, act2.get_shape().as_list()[3]],
                                  output_channel=self.growth_rate, stride=1, init="xavier",
                                  trainable=True, name=name+"_conv2_"+str(iter+1))
                input = tf.concat([input, conv2], axis=3)
        return input

    def transition_layer(self, input, name):
        with tf.variable_scope(name):
            bn = self.BN(input=input, trainable=self.bol, name=name+"_bn")
            conv = self.Conv(input=bn, filter=[1, 1, bn.get_shape().as_list()[3]],
                             output_channel=4*self.growth_rate, stride=1, init="xavier",
                             trainable=True, name=name+"_conv")
            pool = self.Pool(input=conv, filter=2, stride=2, function="avg", name=name+"_pool")
        return pool

    def feed_dict(self, True, x, y, batch_size):
        batch_idx = np.random.choice(x.shape[0], batch_size, False)
        xs, ys = x[batch_idx], y[batch_idx]
        bool = True
        return {self.x: xs, self.y: ys, self.bol: bool}