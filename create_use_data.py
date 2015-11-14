from numpy import *
from numpy.random import *
def generate_use_data(use_data):
	f.open('use_data.txt')
	for i in xrange(use_data):
		if use_data[i]==True:
			f.write('1')
		else:
			f.write('0')
	f.close()