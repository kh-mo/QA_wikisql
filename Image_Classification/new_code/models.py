import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

flags = tf.app.flags
flags.DEFINE_string("train_path", "C:/Users/user/Desktop/Deep_Learning/Image_Classification/data/train", "Training text file. (required)")
flags.DEFINE_string("test_path", "C:/Users/user/Desktop/Deep_Learning/Image_Classification/data/test", "Testing text file. (required)")
flags.DEFINE_string("save_path", "C:/Users/user/Desktop/Deep_Learning/Image_Classification/alexnet", "Directory to write the model. (required)")
flags.DEFINE_integer("train_epochs", 50, "Number of epochs to train. Each epoch processes the training data once.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 2, "Number of training examples processed per step.")
flags.DEFINE_integer("number_of_label", 2, "Number of label.")
FLAGS = flags.FLAGS

class Options(object):
    def __init__(self):
        self.train_path = FLAGS.train_path
        self.test_path = FLAGS.test_path
        self.save_path = FLAGS.save_path
        self.learning_rate = FLAGS.learning_rate
        self.train_epochs = FLAGS.train_epochs
        self.batch_size = FLAGS.batch_size
        self.number_of_label = FLAGS.number_of_label

class Block():
    def variable_initializer(self, init="", filter=[], output_channel=0, trainable=True, name=""):
        with tf.variable_scope(name):
            if name[-4:] == "bias":
                variable = tf.Variable(tf.constant(0.1, shape=[output_channel]))
            else:
                shape = filter + [output_channel]
                if init == "norm":
                    variable = tf.truncated_normal(shape, mean=0, stddev=1, name=name)
                if init == "xavier":
                    variable = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
            return variable

    def Conv(self, input, filter, output_channel, stride, init, trainable, name):
        with tf.variable_scope(name):
            filt = self.variable_initializer(init, filter, output_channel, trainable, name+"_filt")
            bias = self.variable_initializer(output_channel=output_channel, trainable=trainable, name=name+"_bias")
            conv = tf.nn.conv2d(input, filt, [1,stride,stride,1], padding="SAME") + bias
            return conv

    def Act(self, input, function, name):
        if function == "relu":
            act = tf.nn.relu(input, name=name)
            return act

    def Pool(self, input, filter, stride, function, name):
        with tf.variable_scope(name):
            if function == "max":
                pool = tf.nn.max_pool(input, ksize=[1,filter,filter,1], strides=[1,stride,stride,1], padding="SAME")
            elif function == "avg":
                pool = tf.nn.avg_pool(input, ksize=[1,filter,filter,1], strides=[1,stride,stride,1], padding="SAME")
            return pool

    def BN(self, input, trainable, name):
        batch = tf.layers.batch_normalization(input, momentum=0.9, epsilon=0.001, training=trainable, name=name)
        return batch

    def FC(self, input, output_channel, init, trainable, name):
        shape = 1
        for i, j in enumerate(input.get_shape().as_list()):
            if i > 0:
                shape *= j
        weight = self.variable_initializer(init, [shape], output_channel, trainable, name + "_filt")
        bias = self.variable_initializer(output_channel=output_channel, trainable=trainable, name=name + "_bias")
        flat = tf.reshape(input, [-1, shape])
        fc = tf.matmul(flat, weight) + bias
        return fc

class data_batch():
    def Train_generator(self, path):
        image = list()
        label = list()
        for folder in os.listdir(path):
            subpath = path + "/" + folder
            for file in os.listdir(subpath):
                subfile = subpath + "/" + file
                image.append(imresize(imread(subfile), [224, 224, 3]).astype(np.float) / 255.)
                if folder == "man":
                    label.append([1, 0])
                else:
                    label.append([0, 1])
        image = np.stack(image, axis=0)
        label = np.stack(label, axis=0)
        return (image, label)

    def Test_generator(self, path):
        image = list()
        for folder in os.listdir(path):
            subfile = path + "/" + folder
            image.append(imresize(imread(subfile), [224, 224, 3]).astype(np.float) / 255.)
        image = np.stack(image, axis=0)
        return image

class AlexNet(Block, data_batch):
    def __init__(self, options, session):
        self._options = options
        self.train_x, self.train_y = self.Train_generator(self._options.train_path)
        self.test_x = self.Test_generator(self._options.test_path)
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
        self.y = tf.placeholder(tf.float32, shape=[None, self._options.number_of_label], name="output")
        self.bol = tf.placeholder(tf.bool, name="bol")

        self.conv1 = self.Conv(input=self.x, filter=[11,11,3], output_channel=48, stride=4, init="xavier", trainable=True, name="conv1")
        self.pool1 = self.Pool(input=self.conv1, filter=2, stride=1, function="max", name="pool1")

        self.conv2 = self.Conv(input=self.pool1, filter=[5,5,48], output_channel=128, stride=1, init="xavier", trainable=True, name="conv2")
        self.pool2 = self.Pool(input=self.conv2, filter=2, stride=1, function="max", name="pool2")

        self.conv3 = self.Conv(input=self.pool2, filter=[3,3,128], output_channel=192, stride=1, init="xavier", trainable=True, name="conv3")
        self.conv4 = self.Conv(input=self.conv3, filter=[3,3,192], output_channel=192, stride=1, init="xavier", trainable=True, name="conv4")
        self.conv5 = self.Conv(input=self.conv4, filter=[3,3,192], output_channel=128, stride=1, init="xavier", trainable=True, name="conv5")
        self.pool3 = self.Pool(input=self.conv5, filter=2, stride=1, function="max", name="pool3")

        self.fc1 = self.FC(input=self.pool3, output_channel=1024, init="xavier", trainable=True, name="fc1")
        self.fc2 = self.FC(input=self.fc1, output_channel=1024, init="xavier", trainable=True,name="fc2")
        self.fc3 = self.FC(input=self.fc2, output_channel=self._options.number_of_label, init="xavier", trainable=True,name="fc3")
        self.probability = tf.nn.softmax(self.fc3)

        ## cost, accuracy, train 지정
        with tf.variable_scope("cost"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc3))
            tf.summary.scalar("cross_entropy", self.cross_entropy)

        with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fc3, 1), tf.argmax(self.y, 1)), dtype=tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

        ema_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope("training"):
            with tf.control_dependencies(ema_op):
                self.train_step = tf.train.AdamOptimizer(self._options.learning_rate).minimize(self.cross_entropy)

    def feed_dict(self, trainable, x, y, batch_size):
        batch_idx = np.random.choice(x.shape[0], batch_size, False)
        xs, ys = x[batch_idx], y[batch_idx]
        bool = trainable
        return {self.x: xs, self.y: ys, self.bol: bool}

    def train(self, sess):
        sess.run(tf.global_variables_initializer())
        iteration = self._options.train_epochs * round(self.train_x.shape[0] / self._options.batch_size)
        for i in range(iteration):
            _, acc, loss = sess.run([self.train_step, self.accuracy, self.cross_entropy],
                                    self.feed_dict(True, self.train_x, self.train_y,  self._options.batch_size))
            if i % 10 == 1:
                print("iter :", i, "acc :", acc, "loss :", loss)
        return print("training_done!\n acc :", acc, "loss :", loss)

    def save(self, sess):
        tf.train.Saver().save(sess, self._options.save_path)
        return print("Save Done.")

class VggNet(Block):
    def __init__(self):
        print("아직 미완성입니다.")

class GoogleNet(Block):
    def __init__(self):
        print("아직 미완성입니다.")

class ResNet(Block):
    def __init__(self):
        print("아직 미완성입니다.")

class DenseNet(Block):
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
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc))
            tf.summary.scalar("cross_entropy", self.cross_entropy)

        with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fc, 1), tf.argmax(self.y, 1)), dtype=tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

        ema_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope("training"):
            with tf.control_dependencies(ema_op):
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)

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

    def feed_dict(self, trainable, x, y, batch_size):
        batch_idx = np.random.choice(x.shape[0], batch_size, False)
        xs, ys = x[batch_idx], y[batch_idx]
        bool = trainable
        return {self.x: xs, self.y: ys, self.bol: bool}

# a = DenseNet(number_of_label=2, growth_rate=32)
# tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

def main(_):
    opts = Options()
    with tf.Session() as sess:
        # sess = tf.InteractiveSession()
        model = AlexNet(opts, sess)
        model.train(sess)
        model.save(sess)

if __name__ == "__main__":
    tf.app.run()




