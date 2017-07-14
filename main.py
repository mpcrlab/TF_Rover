from RoverRunStack import *
from RoverRun import *
from RoverSimple import *
if __name__ == '__main__':
	a = raw_input('a for autonomous, k for keyboard')
        if a in ['a', 'A']:
            b = raw_input('Stack or no stack (y/n)? ')
	    if b in ['Y', 'y']:
                rover = RoverRun(framestack={'f_int0':1, 'f_int1':4})
            else:
                rover = RoverRun()
        else:
            rover = RoverSimple()
