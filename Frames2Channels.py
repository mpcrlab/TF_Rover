get_ipython().magic(u'matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt
import imageio
import json
from numpy.matlib import *

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


filename = 'drive.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')

nframes = 500

Data = np.zeros((nframes, 480, 640, 3))

nums = range(nframes)

for num in nums:
    Data[num,:,:,:] = vid.get_data(num)


# In[10]:

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.ion()

# fig.show()
# fig.canvas.draw()

# for i in range(nframes):
#     ax.clear()
#     ax.imshow(rgb2gray(-Data[i,:,:,:]))
#     fig.canvas.draw()


Data.shape

t=5

Data_t=np.zeros((Data.shape[0]-t,Data.shape[1],Data.shape[2],t))

for i in range(Data.shape[0]-t):
    print(i)
    
    frame_stack=np.zeros((Data.shape[1],Data.shape[2],t))
    
    for j in range(t):
    
        frame_stack[:,:,j]=rgb2gray(Data[i-j,:,:,:])
    
    Data_t[i,:,:,:]=frame_stack

Data_t.shape
