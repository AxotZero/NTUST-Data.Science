import argparse
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable

def get_args():

	args = {}
	args['epoch'] = 1
	args['learning_rate'] = 0.00025
	args['momentum'] = 0.9
	args['save_dir'] = 'snapshots'

	return args

def get_parser():
	parser = argparse.ArgumentParser(description="Basic setting for Gestures Classifier")

	parser.add_argument("--learning_rate", type=float, default=0.00025,
                        help="Base learning rate for training with polynomial decay.")
	
	parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")

	parser.add_argument("--epoch", type=int, default=1,
						help="Epoch")

	parser.add_argument("--snapshot_dir", type=str, default='snapshots',
                        help="Where to save snapshots of the model.")

	return parser.parse_args()

def get_criterion():
	return nn.CrossEntropyLoss()

def get_optimizer(model,lr,momentum):
	return optim.SGD(model.parameters(), lr=lr, momentum=momentum)

if __name__ == '__main__':
	# utils unit test
	args = get_parser()
	print("Total Epoch : ",args.epoch)
	print("Test OK!!")
