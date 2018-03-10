import argparse
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import matplotlib.pyplot as plt



def mat2ten(X):
    zs=[X.shape[1], int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))]
    Z=np.zeros(zs)

    for i in range(X.shape[1]):
        Z[i, ...] = X[:,i].reshape([zs[1],zs[2]])

    return Z



def montage(X):
    X = mat2ten(X)
    count, m, n = X.shape
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n] = X[image_id, ...]
            image_id += 1
    return M




def plot(x):
    x = montage(x)
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set_size_inches(18, 18)
    plt.show()



parser = argparse.ArgumentParser(description='model to load')
parser.add_argument('file_path', type=str)
args = parser.parse_args()
f = args.file_path

reader = pywrap_tensorflow.NewCheckpointReader(f)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    weights = reader.get_tensor(key)
    shp = weights.shape
    print(shp)
    if 'Conv2D/W' in key and len(shp) == 4:
        for i in range(shp[-1]):
            weights = weights.reshape([shp[0]*shp[1], -1])
            plot(weights)
