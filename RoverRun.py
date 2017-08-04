from __future__ import print_function
import pygame
from Data import *
import pygame.camera
from pygame.locals import *
from NetworkRun import *
from Pygame_UI import *
from rover import Rover
import cv2
import numpy as np
import time
import math
import numpy as np
import h5py
import tflearn
import matplotlib.pyplot as plt
import scipy.misc
import math
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()
import os


class RoverRun(Rover):

    def __init__(self, framestack=False, film=False):
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
        self.timeStart = time.time()
        self.stack = framestack 
	self.film = film
	if self.film is True:
	    pygame.camera.init()
            camlist = pygame.camera.list_cameras()
	    if camlist:
	        self.cam = pygame.camera.Camera(camlist[0],(640,480))
	        self.cam.start()

        if framestack is False:
	    self.network = input_data(shape=[None, 130, 320, 1])
        else:
            self.network = input_data(shape=[None, 130, 320, len(framestack)+1])
            self.framestack = np.zeros([1, 130, 320, self.FPS])
	    self.stack.append(0)
            self.stack.sort()

	self.network = DNN1(self.network)
	self.model = tflearn.DNN(self.network)
	self.model.load('/home/TF_Rover/RoverData/Felix_3frames10-20_FeatureScaling_DNN1')
	self.run()


    def film_run(self):
        return pygame.surfarray.array3d(pygame.transform.rotate(self.cam.get_image(), 90))
		
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

        while type(self.image) == type(None):
            pass


        while not self.quit:
            

            
       	    key = self.getActiveKey()
            if key:
                key = chr(key)

            if key == 'z':
                self.quit = True

	    s=self.image
	
	    if self.film is True:
	        a = self.film_run()
	        cv2.imshow('webcam', a)
	
	    # grayscale and crop
	    s=np.mean(s[None,110:,:,:], 3, keepdims=True)
	
            # Local Feature Scaling
	    s = (s-np.mean(s))/(np.std(s)+1e-6)
            
            # Framestack 
            if self.stack is not False:
                current = s
		self.framestack = np.concatenate((current, self.framestack[:, :, :, 1:]), 3)
		s = self.framestack[:, :, :, self.stack]
                #for i in range(len(self.stack)):
                #    frame = self.framestack[self.stack['f_int{0}'.format(i)], :, :, :]
                #    s = np.concatenate((s, frame[None, :, :, :]), 3)
	    
            self.angle = self.model.predict(s)
	    self.angle = np.argmax(self.angle)
	    
	    
	    os.system('clear')
	    print(self.angle)
            print(self.image.shape)	

	    speed=0.5

            if self.angle == 0:
                self.treads = [-speed,speed]
            elif self.angle == 1:
               self.treads = [speed, speed]
            elif self.angle == 2:
               self.treads = [speed,-speed]


	    self.set_wheel_treads(self.treads[0],self.treads[1])

	    
        
            cv2.imshow("RoverCam", scipy.misc.bytescale(np.mean(self.image[110:, ...], 2)))
	    cv2.waitKey(1)
             
            self.clock.tick(self.FPS)
            pygame.display.flip()
            self.userInterface.screen.fill((255,255,255))


        self.set_wheel_treads(0,0)
        
        pygame.quit()
        cv2.destroyAllWindows()
        self.close()


