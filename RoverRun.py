from __future__ import print_function
import pygame
import glob
from Data import *
import pygame.camera
from pygame.locals import *
import os, sys
from Pygame_UI import *
from rover import Rover
import cv2
import time
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.misc
sys.argv += [1.0]
from NetworkSwitch import *
import tensorflow as tf
tf.reset_default_graph()


class RoverRun(Rover):

    def __init__(self, learn=False):
        Rover.__init__(self)
        self.d = Data()
        self.userInterface = Pygame_UI()
        self.clock = pygame.time.Clock()
        self.FPS = 30
        self.image = None
        self.quit = False
        self.paused = False
        self.angle = 0
        self.treads = [0,0]
	self.learn = learn
        self.timeStart = time.time()
	
	fileName = glob.glob('/home/TF_Rover/RoverData/*.index')
	fileName = fileName[0]
	
	if '1frame_Color' in fileName:
	    self.channs = 3
	    self.im_method = 0
	elif '3frames' in fileName:
	    self.channs = 3
	    self.im_method = 1
	    self.framestack = np.zeros([1, 130, 320, self.FPS])
	    self.stack = [0, 5, 15]
        elif '1frame_Gray' in fileName:
	    self.channs = 1
	    self.im_method = 2
		
	if self.learn:
	    self.n_data = np.zeros([3000, 130, 320, 3])
	    self.n_labels = np.zeros([3000, 1])

	self.network = input_data(shape=[None, 130, 320, self.channs])

	modelFind = fileName[fileName.find('_', 64, len(fileName))+1:-6]
	self.network = globals()[modelFind](self.network)
	self.model = tflearn.DNN(self.network)
	self.model.load(fileName[:-6],
		       weights_only=True)
	self.run()


		
    def getActiveKey(self):
        key = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key = event.key
        return key       



    def process_video_from_rover(self, jpegbytes, timestamp_10msec):
        array_of_bytes = np.fromstring(jpegbytes, np.uint8)
        self.image = cv2.imdecode(array_of_bytes, flags=3)
        k = cv2.waitKey(5) & 0xFF
        return self.image



    def eraseFrames(self, count):
        size = len(self.d.angles)
        if (size - count > 0):
            print("--", "Deleting" , count, "seconds of frames!")
            self.d.angles = self.d.angles[:size - count]
            self.d.images = self.d.images[:size - count]
        else:
            print("Couldn't delete! List has less than", count, "frames!")



    def run(self):
	
	start_time = time.time()

        while type(self.image) == type(None):
            pass


        while not self.quit:
            
	    i = 0
            
       	    key = self.getActiveKey()
            if key:
                key = chr(key)

            if key == 'z':
                self.quit = True

	    s=self.image
	
	    self.n_data[i, ...] = s
	    
	    s=s[None,110:,:,:]
	
	    if self.im_method in [1, 2]:
	        s = np.mean(s, 3, keepdims=True)
	
            # Local Feature Scaling
	    s = (s-np.mean(s))/(np.std(s)+1e-6)
            
            # Framestack 
            if self.im_method == 1:
                current = s
		self.framestack = np.concatenate((current, self.framestack[:, :, :, 1:]), 3)
		s = self.framestack[:, :, :, self.stack]
	    
	    # predict the correct steering angle from input
            self.angle = np.argmax(self.model.predict(s))
	    #self.angle = np.argmax(self.angle)
	    self.n_labels[i, ...] = self.angle - 1
	    
	    os.system('clear')
	    print(self.angle)
            print(self.image.shape)	

	    speed=.5

            if self.angle == 0:
                self.treads = [-(speed - 0.07),speed - 0.07]
            elif self.angle == 1:
               self.treads = [speed, speed]
            elif self.angle == 2:
               self.treads = [speed - 0.07,-(speed - 0.07)]
	    elif self.angle == 3:
		self.treads = [-speed, -speed]


	    self.set_wheel_treads(self.treads[0],self.treads[1])

	    
        
            #cv2.imshow("RoverCam", scipy.misc.bytescale(np.mean(self.image[110:, ...], 2)))
	    #cv2.waitKey(1)
             
            self.clock.tick(self.FPS)
            pygame.display.flip()
            self.userInterface.screen.fill((255,255,255))
	
	    i += 1
	    
	    if i == self.n_labels.shape[0] - 1:
		self.quit == True
	
	elapsed_time = np.round(time.time() - start_time, 2)
	print('This run lasted %.2f seconds'%(elapsed_time))
	
        self.set_wheel_treads(0,0)
	
	nfile = h5py.File('New_Data' + str(time.time()) + '.h5', 'a')
	nfile.create_dataset('X', data=self.n_data)
	nfile.create_dataset('Y', data=self.n_labels)
	nfile.close()
        
        pygame.quit()
        cv2.destroyAllWindows()
        self.close()
