'''
Created on Jul 14, 2015

@author: kashefy
'''
import os
import sys
import caffe
# import cv2 as cv2
# import cv2.cv as cv

from pylab import *

from caffe import layers as L
from caffe import params as P

def lenet(lmdb, batch_size):
	# our version of LeNet: a series of linear and simple nonlinear transformations
	n = caffe.NetSpec()
	n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
							 transform_param=dict(scale=1./255), ntop=2)
	n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.ip1, in_place=True)
	n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
	n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
	return n.to_proto()
def cifar10_net(lmdb, batch_size):
	# our version of LeNet: a series of linear and simple nonlinear transformations
	n = caffe.NetSpec()
	n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
							 transform_param=dict(scale=1./255), ntop=2)
	n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=32, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.conv1, in_place=True)
	n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
	
	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=32, weight_filler=dict(type='xavier'))
	n.relu2 = L.ReLU(n.conv2, in_place=True)
	n.pool2 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

	n.ip1 = L.InnerProduct(n.pool2, num_output=512, weight_filler=dict(type='xavier'))
	n.relu3 = L.ReLU(n.ip1, in_place=True)
	n.ip2 = L.InnerProduct(n.relu3, num_output=10, weight_filler=dict(type='xavier'))
	n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
	return n.to_proto()
	
if __name__ == '__main__':
	
	# os.chdir('/home/kashefy/src/caffe/')
	os.chdir('/home/jianqiao/Caffe/caffe-master/')
	test_batchsz=100
	train_batchsz=200
	with open('examples/cifar10/lenet_auto_train.prototxt', 'w') as f:
		f.write(str(cifar10_net('examples/cifar10/cifar10_train_lmdb', train_batchsz)))
		
	with open('examples/cifar10/lenet_auto_test.prototxt', 'w') as f:
		f.write(str(cifar10_net('examples/cifar10/cifar10_test_lmdb', test_batchsz)))
		
	import lmdb
	
	caffe.set_mode_gpu()
	caffe.set_device(0) # for gpu mode
	# caffe.set_mode_cpu()
	solver = caffe.SGDSolver('examples/cifar10/lenet_cifar10_solver.prototxt')
	
#	 solver.net.forward()  # train net
#	 solver.test_nets[0].forward()  # test net (there can be more than one)
# 
#	 imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
#	 print solver.test_nets[0].blobs['label'].data[:8]

#	 niter = 10
#	 for it in xrange(niter):
#		 print it
#		 solver.step(500)
#		 print "step"
#	 
#		 imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
#			.transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray')
#		 show()

	niter = 206
	test_interval = 20
	# losses will also be stored in the log
	train_loss = zeros(niter)
	test_acc = zeros(int(np.ceil(niter / test_interval))+1)
	print "test_acc",test_acc.shape
	# output = zeros((niter, 8, 10))
	
	filt_hist = []
	net=solver.net
	# net.layers[8].use_data.reshape(1,1,1,train_batchsz)
	# net.layers[7].use_data.reshape(1,1,1,train_batchsz-2)
	# use_data=ones(train_batchsz)
	# use_data[:5]=0.0
	# # use_data.fill(0.0)
	# net.layers[8].use_data.diff[...]=use_data
	
	# use_data_ind=(use_data==0)
	# the main solver loop
	import lmdb
	TOTAL_NUM=400
	use_data=zeros(TOTAL_NUM,dtype=int)
	labelset=zeros(TOTAL_NUM,dtype=int)
	start_index=0
	end_index=start_index + train_batchsz
	LMDB_PATH = "/examples/cifar10/cifar10_train_lmdb/"

	env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
	datum = datum_pb2.Datum()
	with env.begin() as txn:
		cur = txn.cursor()
		for i in xrange(TOTAL_NUM):
			if not cur.next():
				cur.first()
			# Read the current cursor
			key, value = cur.item()
			# convert to datum
			datum.ParseFromString(value)
			labelset[i]=datum.label
			# Read the datum.data
			# img_data = numpy.array(bytearray(datum.data))\
			# 	.reshape(datum.channels, datum.height, datum.width)
	print "labelset",labelset
	for it in range(niter):
		# print "prev",solver.net.blobs['label'].data[...]
		# solver.net.blobs['label'].data[...]=ones(10)
		# print "post",solver.net.blobs['label'].data[...]

		# print "before forward",solver.net.blobs['label'].data[:5]
		# solver.net.forward(start='data',end='conv1')
		# solver.net.forward()
		# solver.net.backward()
		# print it,"prev zeros ",absolute(net.blobs['ip2'].diff).mean()
		# # print net.blobs['ip2'].diff.shape
		# net.blobs['ip2'].diff[...]=0.0
		# print it,"post zeros ",absolute(net.blobs['ip2'].diff).mean()
		# print "prev forward ",absolute(net.blobs['ip2'].diff).mean()
		# print "after forward",solver.net.blobs['label'].data[:5]
		solver.step_forward()
		# solver.net.forward(start='conv1')
		
		# print "data.shape",net.layers[8].use_data.data.shape
		# print "diff.shape",net.layers[8].use_data.diff.shape
		# print "ip2 data.shape",net.blobs['ip2'].data.shape
		# print "ip2 diff.shape",net.blobs['ip2'].diff

		# net.blobs['ip2'].diff[use_data_ind]=0.0
		# if it%2==0:
		# print "after step_forward",solver.net.blobs['label'].data[:5]
		test_=0
		if test_:
			net.blobs['ip2'].diff.fill(0.0)
			net.blobs['ip1'].data.fill(0.0)
			# print net.blobs['ip2'].data.shape
			# print net.blobs['ip1'].data.shape
			print "prev",net.blobs['ip2'].diff.sum(1)[:10]
			layer=net.layers[9]
			print layer.blobs[0].data.shape
			net.backward(diffs=['ip1','ip2'])
			print "post",net.blobs['ip2'].diff.sum(1)[:10]
			tmp_blob=net.layers[4].blobs[0]
			print "prev backward",tmp_blob.data.mean(),tmp_blob.diff.mean()

		# print net.layers[7].use_data.data.shape
		# print net.layers[8].use_data.count
		# print it,"post forward ",absolute(net.blobs['ip2'].diff).mean()
		# print net._layer_names[8]
		# if it>0:
		# 	print net.layers[8].prob_.shape
		# print net.layers[0].blobs[0].data.shape
		# print "conv1 shape",net.blobs['conv1'].shape
		# print "conv1 data",net.blobs['conv1'].data[:10,0,0,0]
		# print "conv1 diff",net.blobs['conv1'].diff[:10,0,0,0]
		
		solver.step_backward()
		# tmp_blob.diff.fill(0.0)
		# print "post backward",tmp_blob.data.mean(),tmp_blob.diff.mean()

		# print "conv1 data",net.blobs['conv1'].data[:10,0,0,0]
		# print "conv1 diff",net.blobs['conv1'].diff[:10,0,0,0]
		# print "ip1",net.blobs['ip1'].diff[:10,:,:,:].sum(1).sum(1).sum(1)
		# net.blobs['ip2'].diff[...]=0.0
		# net.blobs['ip2'].diff[...]=0.0
		# net.blobs['ip1'].data[...]=0.0
		solver.apply_update()
		
		# print "post update",tmp_blob.data.mean(),tmp_blob.diff.mean()
		solver.step_extra()
		# print "after step_extra",solver.net.blobs['label'].data[:5]
		# print " "
		# print it,"post backward ",absolute(net.blobs['ip2'].data).mean()
		# solver.step(1)  # SGD by Caffe
		# print solver.test_nets[0].params['conv1'][0].data[1,0,:,:]
		
		# store the train loss
		train_loss[it] = solver.net.blobs['loss'].data
		# print "final",solver.net.blobs['label'].data
		# store the output on the first test batch
		# (start the forward pass at conv1 to avoid loading new data)
		# solver.test_nets[0].forward(start='conv1')
		# diffs = solver.net.backward(diffs=['data','conv1'])
		# print "diffs.shape",solver.net.blobs['data'].diff.shape
		# output[it] = solver.test_nets[0].blobs['ip2'].data[:8]
		
		# run a full test every so often
		# (Caffe can also do this for us and write to a log, but we show here
		#  how to do it directly in Python, where more complicated things are easier.)
		
		start_index =(start_index + train_batchsz) % TOTAL_NUM
		if start_index + train_batchsz>TOTAL_NUM:
			tmp=TOTAL_NUM - start_index
			end_index=train_batchsz - tmp
			label[:tmp]= labelset[start_index:TOTAL_NUM]
			label[tmp:]= labelset[:end_index]
		else:
			end_index= start_index + train_batchsz
			label[:] = labelset[start_index:end_index]

		if it % test_interval == 0 and it>0:
			print 'Iteration', it, 'testing...'
			correct = 0
			test_iters=10
			for test_it in range(test_iters):
				solver.test_nets[0].forward()
				correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
							   == solver.test_nets[0].blobs['label'].data)
			test_acc[it // test_interval] = float(correct) / (test_iters * test_batchsz)
			
			# filt_hist.append(solver.test_nets[0].params['conv1'][0].data)


	# plot_=1
	# if plot_:	   
	fig = figure(1)
	# weights = filt_hist[-1]
	# n = int(np.ceil(np.sqrt(weights.shape[0])))
	# for i, f in enumerate(weights):
	#	 ax = fig.add_subplot(n, n, i+1)
	#	 ax.axis('off')
	#	 cm = None
	#	 if f.ndim > 2 and f.shape[0]==1:
	#		 f = f.reshape(f.shape[1:])
	#	 if f.ndim == 2 or f.shape[0]==1:
	#		 cm = 'gray'
	#	 imshow(f, cmap=cm)
	# show()
		
	_, ax1 = subplots()
	ax2 = ax1.twinx()
	ax1.plot(arange(niter), train_loss)
	ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	ax2.set_ylabel('test accuracy')
	
	show()
	
	# n = int(np.ceil(np.sqrt(d.shape[0])))
	# fig = figure()
	# for i, f in enumerate(d):
	#	 ax = fig.add_subplot(n, n, i+1)
	#	 ax.axis('off')
	#	 print f.shape
	#	 cm = None
	#	 if f.ndim > 2 and f.shape[0]==1:
	#		 f = f.reshape(f.shape[1:])
	#	 if f.ndim == 2 or f.shape[0]==1:
	#		 cm = 'gray'
	#	 imshow(f, cmap=cm)
			
	
	pass