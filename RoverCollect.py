from __future__ import print_function
import pygame
from Controller import *
from Data import *
from Pygame_UI import *
from rover import Rover
import cv2
import numpy as np
import time
import math

class RoverSimple(Rover):
    def __init__(self):
        Rover.__init__(self)
        self.d = Data()
        self.userInterface = Pygame_UI()
        self.clock = pygame.time.Clock()
        self.FPS = 30
        self.image = None
        self.quit = False
        self.controller = None
        self.controllerType = None
        self.paused = False
        self.angle = None
        self.treads = [0,0]
        self.timeStart = time.time()
        self.run()

    def setControls(self):
    	self.controllerType = "Keyboard"
   	self.paused = True
    	self.controller = Keyboard()
           

    def reverse(self):
        self.treads = [-1,-1]

    def freeze(self):
        self.treads = [0,0]
        self.set_wheel_treads(0,0)

 

    def process_video_from_rover(self, jpegbytes, timestamp_10msec):
        window_name = 'Machine Perception and Cognitive Robotics'
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



    def displayDashboard(self):
        black = (0,0,0)

        self.userInterface.display_message("Rover Battery Percentage: " + str(self.get_battery_percentage()), black, 0,0)
        self.userInterface.display_message("Treads: " + str(self.treads), black, 0, self.userInterface.fontSize*4)
        self.userInterface.display_message("Number of Frames Collected: " + str(len(self.d.angles)), black, 0, self.userInterface.fontSize*7)



    def run(self):
        while type(self.image) == type(None):
            pass

        oldTreads = None
        self.setControls()


        while not self.quit:
            self.displayDashboard()

            
       	    key = self.controller.getActiveKey()
            if key:
                key = chr(key)
            if key in ['w','a','d']:
                self.angle = self.controller.getAngle(key)
                self.paused = False
            elif key == 'z':
                self.quit = True
            elif key == 'b':
                print(self.get_battery_percentage())
            elif key == ' ':
                self.paused = not self.paused
            elif key == 'p':
                self.eraseFrames(self.FPS)


	    if self.paused:
                self.freeze()



            if self.angle == 135:
                self.treads = [-1,1]
            elif self.angle == 90:
               self.treads = [1, 1]
            elif self.angle == 35:
               self.treads = [1,-1]



           

	    newTreads = self.treads
	    self.set_wheel_treads(newTreads[0],newTreads[1])



            self.d.angles.append(self.angle)
            self.d.images.append(self.image)
           
            # Displaying images 
            cv2.imshow("RoverCam", self.image)
             
        

            self.clock.tick(self.FPS)
            pygame.display.flip()
            self.userInterface.screen.fill((255,255,255))



        self.set_wheel_treads(0,0)
        self.d.save('dataset.h5')
        pygame.quit()
        cv2.destroyAllWindows()
        self.close()
