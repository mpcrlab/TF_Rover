import argparse
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.misc import bytescale, imresize



def montage(X):
    ps = X.shape[0]

    if X.shape[2] == 3:
        n = np.sqrt(X.shape[-1])
        out = np.zeros([int(np.ceil(ps*n)), int(np.ceil(ps*n)), 3])
        n = int(n)
        
        for i in range(n):
            for j in range(n):
                out[i*ps:i*ps+ps, j*ps:j*ps+ps, :] = X[..., i*n+j]

    else:
        out = np.zeros([ps*X.shape[2], ps*X.shape[3]])
        print(out.shape)
        for i in range(X.shape[2]):
            for j in range(X.shape[3]):
                out[i*ps:i*ps+ps, j*ps:j*ps+ps] = X[..., i, j]

    return out



def plot(x, name, color=True):
    x = montage(x)
    fig, ax = plt.subplots()
    if color:
        im = ax.imshow(x)
    else:
        im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(18, 18)
    plt.title(name)
    plt.show()



parser = argparse.ArgumentParser(description='model to load')
parser.add_argument('file_path', type=str)
args = parser.parse_args()
f = args.file_path

reader = pywrap_tensorflow.NewCheckpointReader(f)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    weights = reader.get_tensor(key)
    shp = weights.shape
    print(shp)

    if 'Conv2D' in key and len(shp) == 4 and 'W/' not in key:
        if 'Color' in f and key == 'Conv2D/W':
            plot(weights, key + ' ' + str(shp))
        else:
            plot(weights, key + ' ' + str(shp), color=False)
