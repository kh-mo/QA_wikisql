import os
import numpy as np
from scipy.misc import imread, imresize

def Train_generator(path):
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

def Test_generator(path):
    image = list()
    for folder in os.listdir(path):
        subfile = path + "/" + folder
        image.append(imresize(imread(subfile), [224, 224, 3]).astype(np.float) / 255.)
    image = np.stack(image, axis=0)
    return image
