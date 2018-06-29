from RoverRun import *
from RoverSimple import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
    '--model_name',
    type=str,
    default='',
    help='name of the model file')
        
    parser.add_argument(
    '--autonomous',
    type=str,
    default='y',
    help='Autonomous or human-controlled? (y/n)')

        args = parser.parse_args()
	a = args.autonomous
        model_file_name = args.model_name

        if a in ['y', 'Y']:
            rover = RoverRun(filename=model_file_name)
        else:
            rover = RoverSimple()
