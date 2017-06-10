import time
import numpy as np
import h5py
import progressbar
import datetime

class Data():
    def __init__(self):
        self.angles = []
        self.images = []

    def load(self):
        pass

    def save(self):

        self.images = np.array(self.images, dtype='uint8')

        self.angles = np.array(self.angles, dtype='float16')

        save_time = datetime.datetime.now().strftime("%m%d%H%M%S")

        #dset_name = "Run_" + save_time + '.h5'
	dset_name ='test2.h5'

	h5f = h5py.File(dset_name, 'w')
	h5f.create_dataset('X', data=self.images)
	h5f.create_dataset('Y', data=self.angles)

	h5f.close()

