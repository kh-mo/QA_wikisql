import tensorflow as tf

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
