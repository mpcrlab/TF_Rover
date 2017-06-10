import numpy as np
import h5py
import tflearn
import matplotlib.pyplot as plt
import scipy.misc
import math



h5f = h5py.File('test.h5', 'r')


print(h5f.keys())



X = np.asarray(h5f['X'])
Y1 = np.asarray(h5f['Y'])+1
Y1 = Y1.astype(int)


print(X.shape)
print(Y1.shape)

print(Y1)

print(np.max(Y1)+1)

Y=np.zeros((Y1.shape[0],np.max(Y1)+1))

for i in range(Y1.shape[0]):
	Y[i,Y1[i]]=1

print(Y)

nframes=Y1.shape[0]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()

for i in range(nframes):
     print(i)
     ax.clear()
     ax.imshow(X[i,:,:,:])
     fig.canvas.draw()

















