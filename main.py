from RoverRun import *
from RoverSimple import *
if __name__ == '__main__':
	a = raw_input('a for autonomous, k for keyboard')
        if a in ['a', 'A']:
	    rover = RoverRun()
        else:
            rover = RoverSimple()
