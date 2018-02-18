from RoverRun import *
from RoverSimple import *
if __name__ == '__main__':
	a = raw_input('a for autonomous, k for keyboard')
        if a in ['a', 'A']:
	    a2 = raw_input('do you want to learn? (y / n) ')
	    if a2 in ['N', 'n']:
            	rover = RoverRun()
	    else:
		rover = RoverRun(learn=True)
        else:
            rover = RoverSimple()
