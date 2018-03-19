import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

flags = tf.app.flags
flags.DEFINE_string("graph_path", "C:/Users/user/Desktop/Deep_Learning/Image_Classification/model/alexnet.meta", "Directory to write the model. (required)")
flags.DEFINE_string("param_path", "C:/Users/user/Desktop/Deep_Learning/Image_Classification/model/alexnet", "Directory to write the model. (required)")
flags.DEFINE_string("test_path", "C:/Users/user/Desktop/Deep_Learning/Image_Classification/data/test", "Testing text file. (required)")
FLAGS = flags.FLAGS

class Options(object):
    def __init__(self):
        self.graph_path = FLAGS.graph_path
        self.param_path = FLAGS.param_path
        self.test_path = FLAGS.test_path

class data_batch():
    def Test_generator(self, path):
        image = list()
        for folder in os.listdir(path):
            subfile = path + "/" + folder
            image.append(imresize(imread(subfile), [224, 224, 3]).astype(np.float) / 255.)
        image = np.stack(image, axis=0)
        return image

def main():
    opts = Options()
    with tf.Session() as sess:
        sess = tf.Session()
        load_graph = tf.train.import_meta_graph(opts.graph_path)
        load_graph.restore(sess, opts.param_path)
        prob = sess.run(sess.graph.get_operation_by_name('Softmax').outputs[0],
                        feed_dict={sess.graph.get_operation_by_name('input').outputs[0]:
                                       data_batch().Test_generator(opts.test_path)})
        print("남성일 확률 : %g, 여성일 확률 : %g" % (prob[0][0], prob[0][1]))
        # tf.get_default_graph().get_operations()

if __name__ == "__main__":
    main()

