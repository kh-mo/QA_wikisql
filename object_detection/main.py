import os
import sys
sys.path.append(os.getcwd() + "/object_detection/")
import tarfile
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import label_map_util
from matplotlib import pyplot as plt
from utils import visualization_utils as vis_util

# MODEL_FILE = os.getcwd() + "/object_detection/1. mmm/ssd_mobilenet_v1_coco_2017_11_17.tar.gz"
# PATH_TO_CKPT = os.getcwd() + "/object_detection/1. mmm/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
# PATH_TO_LABELS = os.getcwd() + "/object_detection/data/mscoco_label_map.pbtxt"
MODEL_FILE = os.getcwd() + "/object_detection/1. mmm/faster_rcnn_resnet50_coco_2017_11_08.tar.gz"
PATH_TO_CKPT = os.getcwd() + "/object_detection/1. mmm/faster_rcnn_resnet50_coco_2017_11_08/frozen_inference_graph.pb"
PATH_TO_LABELS = os.getcwd() + "/object_detection/data/mmm_label_map.pbtxt"
# PATH_TO_LABELS = os.getcwd() + "/object_detection/data/mscoco_label_map.pbtxt"

tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd() + "/object_detection/1. mmm/")

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='') # Imports the graph from graph_def into the current default Graph

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = os.getcwd() + "/object_detection/test_images"
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 4)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # print(image_tensor)
        # print(detection_boxes)
        # print(detection_scores)
        # print(detection_classes)
        # print(num_detections)
        for image_path in TEST_IMAGE_PATHS:
              image = Image.open(image_path)
              # the array based representation of the image will be used later in order to prepare the
              # result image with boxes and labels on it.
              image_np = load_image_into_numpy_array(image)
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
              image_np_expanded = np.expand_dims(image_np, axis=0)
              image_np_expanded.shape
              # Actual detection.
              (boxes, scores, classes, num) = sess.run(
                                                [detection_boxes, detection_scores, detection_classes, num_detections],
                                                feed_dict={image_tensor: image_np_expanded})

              # Visualization of the results of a detection.
              vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                                    np.squeeze(classes).astype(np.int32),
                                                                    np.squeeze(scores), category_index,min_score_thresh=.7,
                                                                    use_normalized_coordinates=True, line_thickness=8)
              plt.figure(figsize=IMAGE_SIZE)
              plt.imshow(image_np)



#
#
# # show graph
# # tensorboard --logdir=./
# sess = tf.InteractiveSession(graph=detection_graph)
# sess.run(tf.global_variables_initializer())
# train_writer = tf.summary.FileWriter(os.getcwd() + "/object_detection/", sess.graph)
