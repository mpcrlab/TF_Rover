import time
import numpy as np
import h5py
import progressbar
import datetime



class Data():
    def __init__(self):
        self.angles = []
        self.images = []
	self.start = time.time()


    def load(self):
        pass

    def save(self):

        self.images = np.array(self.images, dtype='uint8')

        self.angles = np.array(self.angles, dtype='float16')

        elapsedTime = int(time.time() - self.start)



	
        dset_name = "Run_" + str(elapsedTime) + "seconds_" + "Chris_Sheri" + ".h5"


	h5f = h5py.File(dset_name, 'w')
	h5f.create_dataset('X', data=self.images)
	h5f.create_dataset('Y', data=self.angles)

	h5f.close()


	



