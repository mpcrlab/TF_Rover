from NetworkSwitch import *
import os, sys, glob, h5py
import tflearn
from scipy.misc import bytescale, imshow
import matplotlib.pyplot as plt
import cv2
from progressbar import ProgressBar

num_test = 50

model_loc = '/home/mpcr/TF_Rover/RoverData/Right2' # path to data/models
model_name = os.path.join(model_loc, 'AlexNet_1frame_ColorCropped2_Alex1')
data_name = os.path.join(model_loc, 'Run_206seconds_Michael_Sheri.h5')

labels = list(range(4))
actions = ['left', 'forward', 'right', 'backward']

h5file = h5py.File(data_name, 'r')
X = h5file['X']
Y = np.float32(h5file['Y'])

network = tflearn.input_data([None, 130, 320, 3])
output = modelswitch['AlexNet'](network, 4)
model = tflearn.DNN(output)
model.load(model_name)


locations = np.random.randint(0, X.shape[0], num_test)
locations = np.sort(locations)
x = X[locations, ...]

mean_diff = 0.
num_flipped = 0.

bar = ProgressBar()
#x = cv2.imread('{}.png'.format(label))
x = x[:, 110:, ...]

for x2 in bar(x):
    heatmap = np.zeros([130, 320])

    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            x_drop = x2.copy()

            if i == 0 and j == 0:
                conf = model.predict(((x_drop - np.mean(x_drop)) / (np.std(x_drop) + 1e-8))[None, ...])
                conf_max = np.argmax(conf, 1)

            chann = x_drop[i, j, :]

            for c, cv in enumerate(chann):
                if cv > 123:
                    x_drop[i, j, c] = 0
                else:
                    x_drop[i, j, c] = 255

            x_norm = (x_drop - np.mean(x_drop)) / (np.std(x_drop) + 1e-8)
            x_norm_out = model.predict(x_norm[None, ...])
            out_max = np.argmax(x_norm_out, 1)
            if out_max != conf_max:
                num_flipped += 1

            heatmap[i, j] = np.sum((conf - x_norm_out)**2)

            #if x_norm_out[0, out_max] > conf[0, conf_max]:
            #    print('Pixel {}, {} increased accuracy'.format(i, j))

    mean_diff += np.mean(heatmap)
    #imshow(heatmap)

print('The average number flipped per image was {}'.format(num_flipped / num_test))
print('The average difference was {}'.format(mean_diff / num_test))
    #cv2.imwrite(actions[label] + '.png', x_drop)
    #cv2.imwrite(actions[label] + 'LSTM.png', bytescale(heatmap))

h5file.close()
