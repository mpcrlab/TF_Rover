from RoverRun import *
from RoverSimple import *
import argparse
from NetworkSwitch import *


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

    parser.add_argument(
	'--network',
	type=int,
	help='Number of the network that exists in the model file.')

    args = parser.parse_args()
    a = args.autonomous
    model_file_name = args.model_name
    network_name = args.network

    if a in ['y', 'Y']:
        rover = RoverRun(model_file_name, modelswitch[network_name])
    else:
        rover = RoverSimple()
