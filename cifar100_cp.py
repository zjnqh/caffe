'''
Created on Jul 14, 2015

@author: kashefy
'''
import os
import sys
import caffe
# import cv2 as cv2
# import cv2.cv as cv
import cPickle as pkl
from pylab import *
import numpy.random as rnd
from caffe import layers as L
from caffe import params as P
rnd.seed(100)
RANDOM_VEC=rnd.randn(5)
FEATURE_INDEX=array([30,60,90,120,300])
def create_labelset(data_num,prototxt,label_ratio=0.1):
	solver = caffe.SGDSolver('examples/cifar10/lenet_cifar10_solver.prototxt')
	net=solver.net
	labelset=zeros(data_num,dtype=int)
	cifar_table=dict()
	for it in xrange(TOTAL_NUM/train_batchsz+3):
		start_index=it* train_batchsz
		end_index=(it+1)* train_batchsz 
		solver.step_forward()
		if end_index>TOTAL_NUM:
			labelset[start_index:TOTAL_NUM]=net.blobs['label'].data[:TOTAL_NUM- start_index]
			break
		else:
			labelset[start_index:end_index]=net.blobs['label'].data
		dat=net.blobs['data'].data
		print "label shape",dat.shape[0],dat[0].shape
		# for n in xrange(dat.shape[0]):
		# 	key=str(RANDOM_VEC.dot(dat[n,FEATURE_INDEX]))
		# 	cifar_table[key]=n+ start_index
	pkl.dump((labelset,cifar_table),open('examples/cifar10/labelset.pkl','wb'))
	
	return (labelset,cifar_table)

class KNN():
	def __init__(self,N,dim,yd):
		self.N=N
		self.D=dim
		self.yd=yd
		self.feature_mat=zeros([N,dim])
		self.label_mat=zeros([N,yd])
		self.capacity=0
		self.index=0
	def update(self,x,y):
		y=y.astype(int)
		if x.shape[0]+self.index>self.N:
			end=self.N
		else:
			end=self.index+ x.shape[0]
		rest=end - self.index
		ind=zeros([rest,2],dtype=int)
		ind[:,0]=arange(self.N)[self.index:end]
		ind[:,1]=y[:rest]
		self.label_mat[arange(self.index,end),y[:rest]]=1.0
		self.index=x.shape[0]+self.index
		if x.shape[0]+self.index>self.N:
			self.feature_mat[:x.shape[0]- rest]=x[rest:]
			self.label_mat[:x.shape[0]- rest]=0.0
			self.label_mat[arange(x.shape[0]- rest),y[rest:]]=1.0
			self.index=self.index% self.N
		self.capacity+= x.shape[0]
		if self.capacity>self.N:
			self.capacity=self.N
	def predict(self,x,yp_cnn,y=None,thh=0.25,print_=False):
		U=x.shape[0]
		affmat=zeros([U,self.N])
		sz=16
		num_batch=U/sz
		for i in xrange(num_batch):
			affmat[i*sz:(i+1)*sz]=((
				x[i*sz:(i+1)*sz,newaxis,:]- self.feature_mat[newaxis,:,:])**2).sum(2)
		sigma=affmat.mean()
		# print "affmat ",affmat.mean(),affmat.max(),affmat.min()
		index=argsort(affmat,axis=1)[:,50:]
		w=exp(-affmat/sigma)
		w[arange(U)[:,newaxis],index]=0.0
		yp_np=w.dot(self.label_mat)
		yp_np=yp_np/(yp_np.sum(1)[:,newaxis])

		y2p=exp(yp_cnn/5.0)
		y2p=y2p/(y2p.sum(1)[:,newaxis])
		alpha=0.5
		yp_np+=alpha*(y2p- yp_np)
		yp=yp_np.argmax(1)
		confidence=yp_np.max(1)
		pivot=argsort(confidence)[U * 0.7]
		ind=confidence>(confidence[pivot])
		accuracy=0.0
		if ind.sum()>0:
				# print "yp shape",yp.shape,y.shape,ind.min(),ind.max()
				accuracy=nan_to_num(np.mean(yp[ind]==y[ind])*100)
		if print_:
			# print "confidence mean",yp_np.mean(),yp_np.max(),yp_np.min()
			print "using ",ind.sum()," unlabeled data "," of accuracy ",accuracy
		return yp[ind],ind,accuracy
def cifar100_net(lmdb, batch_size):
	# our version of LeNet: a series of linear and simple nonlinear transformations
	n = caffe.NetSpec()
	n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
							 transform_param=dict(scale=1./255), ntop=2)
	n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=64, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.conv1, in_place=True)
	# n.norm1 = L.LRN(n.relu1, local_size=3, alpha= 0.0001, beta= 0.75)
	n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
	
	n.conv2 = L.Convolution(n.pool1, kernel_size=3, num_output=128, weight_filler=dict(type='xavier'))
	n.relu2 = L.ReLU(n.conv2, in_place=True)
	# n.norm2 = L.LRN(n.relu2, local_size=3, alpha= 0.0001, beta= 0.75)
	n.pool2 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

	n.conv3 = L.Convolution(n.pool2, kernel_size=2, num_output=192, weight_filler=dict(type='xavier'))
	n.relu5 = L.ReLU(n.conv3, in_place=True)

	n.ip0 = L.InnerProduct(n.relu5, num_output=2048, weight_filler=dict(type='xavier'))
	n.relu3 = L.ReLU(n.ip0, in_place=True)
	# n.drop3= L.Dropout(n.relu3,dropout_ratio=0.5)

	n.ip1 = L.InnerProduct(n.relu3, num_output=2048, weight_filler=dict(type='xavier'))
	n.relu4 = L.ReLU(n.ip1, in_place=True)
	# n.drop4= L.Dropout(n.relu4,dropout_ratio=0.5)

	n.ip2 = L.InnerProduct(n.relu4, num_output=100, weight_filler=dict(type='xavier'))
	n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
	return n.to_proto()
	
if __name__ == '__main__':
	
	# os.chdir('/home/kashefy/src/caffe/')
	os.chdir('/home/jianqiao/Caffe/caffe-master/')
	prototxt='examples/cifar100/lenet_cifar100_solver.prototxt'
	test_batchsz=1000
	train_batchsz=256
	with open('examples/cifar100/lenet_auto_train.prototxt', 'w') as f:
		f.write(str(cifar100_net('examples/cifar100/cifar100_train_lmdb', train_batchsz)))
		
	with open('examples/cifar100/lenet_auto_test.prototxt', 'w') as f:
		f.write(str(cifar100_net('examples/cifar100/cifar100_test_lmdb', test_batchsz)))
	
	caffe.set_mode_gpu()
	caffe.set_device(0) # for gpu mode
	# caffe.set_mode_cpu()
	TOTAL_NUM=50000
	# (labelset,cifar_table )=create_labelset(TOTAL_NUM,prototxt)
	# (labelset,cifar_table )=pkl.load(open('examples/cifar10/labelset.pkl','rb'))
	solver = caffe.SGDSolver(prototxt)

	niter = 202
	test_interval = 40
	train_loss = zeros(niter)
	test_acc = zeros(int(np.ceil(niter / test_interval))+1)
	filt_hist = []
	net=solver.net
	start_index=0
	end_index=start_index + train_batchsz

	semi_start=0
	unlabeled=rand(train_batchsz)>0
	knn=KNN(600,1024,100)
	label_ratio=0.2
	use_data=rnd.randn(TOTAL_NUM)<label_ratio
	for it in xrange(niter):
		solver.step_forward()
		# print "after step_forward",solver.net.blobs['label'].data[:5]

		# end_index=start_index + train_batchsz
		# if end_index>TOTAL_NUM:
		# 	end_index=TOTAL_NUM
		# 	unlabeled[:end_index- start_index]=use_data[start_index:end_index]==0
		# 	unlabeled[end_index- start_index:]=use_data[:start_index+ train_batchsz- TOTAL_NUM]==0
		# 	start_index+= train_batchsz- TOTAL_NUM
		# else:
		# 	unlabeled=use_data[start_index:end_index]==0
		# 	start_index= end_index
		# labeled=logical_not(unlabeled)
		# print_=0
		# if it>semi_start:
		# 	knn.update(net.blobs['ip1'].data[labeled],net.blobs['label'].data[labeled])
		# 	if it%4==0:
		# 			print_=1
		# 	new_data_exist=1
		# 	if knn.capacity==knn.N:
		# 		(yp,ind,accuracy)=knn.predict(net.blobs['ip1'].data[unlabeled],
		# 			net.blobs['ip2'].data[unlabeled],
		# 			net.blobs['label'].data[unlabeled],print_=print_)
		# 	else:
		# 		new_data_exist=0
		# 	if new_data_exist:
		# 		new_add_data=arange(train_batchsz)[unlabeled][ind]
		# 		unlabeled[ind]=False
		# 		net.blobs['label'].data[new_add_data]=yp

		# net.blobs['ip2'].diff[unlabeled].fill(0.0)
		# net.blobs['ip1'].data[unlabeled].fill(0.0)
			# if yp!=None:net.blobs['label'].data[new_add_data]=yp

		solver.step_backward()
		
		solver.apply_update()
		
		solver.step_extra()
		
		train_loss[it] = solver.net.blobs['loss'].data
		
		# start_index =(start_index + train_batchsz) % TOTAL_NUM
		# if start_index + train_batchsz>TOTAL_NUM:
		# 	tmp=TOTAL_NUM - start_index
		# 	end_index=train_batchsz - tmp
		# 	label[:tmp]= labelset[start_index:TOTAL_NUM]
		# 	label[tmp:]= labelset[:end_index]
		# else:
		# 	end_index= start_index + train_batchsz
		# 	label[:] = labelset[start_index:end_index]

		if it % test_interval == 0 and it>0:
			print 'Iteration', it, 'testing...'
			correct = 0
			test_iters=5
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
	f=open("cifar100_result.txt","ab")
	f.write(str('-'*80)+"\n")
	f.write("iteration "+str(niter)+"\n")
	f.write("train accuracy "+str(test_acc)+"\n")
	f.write("loss "+str(train_loss[::10])+"\n \n")
	f.close()
	pass