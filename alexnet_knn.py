'''
Created on Jul semi_layer, 2015

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
from scipy.linalg import *
rnd.seed(100)
RANDOM_VEC=rnd.randn(5)
FEATURE_INDEX=array([30,60,90,120,300])
def create_labelset(data_num,prototxt,label_ratio=0.1):
	solver = caffe.SGDSolver('examples/cifar100/lenet_cifar100_solver.prototxt')
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
		# solver.step_backward()
		
		# solver.apply_update()
		
		# solver.step_extra()
		# for n in xrange(dat.shape[0]):
		# 	key=str(RANDOM_VEC.dot(dat[n,FEATURE_INDEX]))
		# 	cifar_table[key]=n+ start_index
	pkl.dump((labelset,cifar_table),open('examples/cifar100/labelset.pkl','wb'))
	
	return (labelset,cifar_table)
def generate_use_data(use_data):
	f=open('use_data.txt',"wb")
	for i in xrange(use_data.shape[0]):
		if use_data[i]==True:
			f.write('1')
		else:
			f.write('0')
	f.close()
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
		# ind=zeros([rest,2],dtype=int)
		# ind[:,0]=arange(self.N)[self.index:end]
		# ind[:,1]=y[:rest]
		self.feature_mat[self.index:end]=x[:rest]
		self.label_mat[self.index:end]=0.0
		# print "self.index ",self.index," end ",end," rest ",rest
		self.label_mat[arange(self.index,end),y[:rest]]=1.0
		self.index=x.shape[0]+self.index
		if self.index>self.N:
			self.feature_mat[:self.index - self.N]=x[rest:]
			self.label_mat[:self.index - self.N]=0.0
			self.label_mat[arange(self.index - self.N),y[rest:]]=1.0
			self.index=self.index% self.N
		self.capacity+= x.shape[0]
		if self.capacity>self.N:
			self.capacity=self.N
		# print "update"
	def predict(self,x,yp_cnn,y=None,thh=0.25,print_=False):
		U=x.shape[0]
		affmat=zeros([U,self.N])

		sz=16
		num_batch=U/sz
		for i in xrange(num_batch):
			affmat[i*sz:(i+1)*sz]=((x[i*sz:(i+1)*sz,newaxis,:]
				- self.feature_mat[newaxis,:,:])**2).mean(2)
		K=30
		sorted_index=argsort(affmat,axis=1)
		index=sorted_index[:,K:]
		affmat[arange(U)[:,newaxis],index]=0.0
		sigma=affmat.sum()/(U* (K)*5)
		# print "affmat ",affmat.mean(),affmat.max(),affmat.min()
		w=exp(-affmat/sigma)
		w[arange(U)[:,newaxis],index]=0.0
		yp_np=w.dot(self.label_mat)
		yp_np=yp_np/(yp_np.sum(1)[:,newaxis])

		y2p=exp(yp_cnn/5.0)
		y2p=y2p/(y2p.sum(1)[:,newaxis])
		alpha=0.7
		yp_np+=alpha*(y2p- yp_np)
		yp=yp_np.argmax(1)
		# yp[:0.5*U]=y[:0.5*U]
		confidence=yp_np.max(1)
		pivot=argsort(confidence)[U * thh]
		ind=confidence>(confidence[pivot])
		accuracy=0.0
		if ind.sum()>0:
				# print "yp shape",yp.shape,y.shape,ind.min(),ind.max()
				accuracy=nan_to_num(np.mean(yp[ind]==y[ind])*100)
		if print_:
			# print "confidence mean",yp_np.mean(),yp_np.max(),yp_np.min()
			print "using ",ind.sum()," unlabeled data "," of accuracy ",accuracy
		return yp[ind],ind,accuracy

if __name__ == '__main__':
	
	# os.chdir('/home/kashefy/src/caffe/')
	os.chdir('/home/jianqiao/Caffe/caffe-master/')
	
	test_batchsz=1000
	train_batchsz=128
	caffe.set_mode_gpu()
	caffe.set_device(0) # for gpu mode
	# caffe.set_mode_cpu()
	TOTAL_NUM=50000
	# (labelset,cifar_table )=create_labelset(TOTAL_NUM, prototxt)
	# (labelset,cifar_table )=pkl.load(open('examples/cifar100/labelset.pkl','rb'))
	
	filt_hist = []
	
	start_index=0
	end_index=start_index + train_batchsz

	unlabeled=rand(train_batchsz)>0
	yd=100
	
	label_ratio=0.1
	rnd.seed(100)
	use_data=rnd.rand(TOTAL_NUM)<label_ratio
	print "python use data :",use_data.sum()
	generate_use_data(use_data)
	import subprocess
	print "start"
	# subprocess.call("./create_cifar_subset.sh")
	print "end"
	# print 'Iteration', it, 'testing accuracy:',float(correct) / (test_iters *test_batchsz)
	knn=KNN(700,640,yd)
	niter = 20003
	semi_start=500
	semi_=1
	test_interval = 50
	train_loss = zeros(niter)
	test_acc = zeros(int(np.ceil(niter / test_interval))+1)
	label=zeros(train_batchsz,dtype=int)
	# prototxt='models/bvlc_alexnet/solver.prototxt'
	prototxt='models/bvlc_alexnet/full_solver.prototxt'
	solverstate='models/bvlc_alexnet/caffe_alexnet_train_iter_1500.solverstate'
	solverstate='models/bvlc_alexnet/solver_state/alexnet_semi_train_iter_2000.solverstate'
	solver = caffe.SGDSolver(prototxt)
	# caffemodel='bvlc_alexnet.caffemodel'
	caffemodel='models/bvlc_alexnet/solver_state/alex_cifar_sub_train_iter_1000.caffemodel'
	# solver.restore(solverstate)
	solver.net.copy_from(caffemodel)
	net=solver.net
	semi_layer=23
	yp=0.0
	net.layers[semi_layer].use_data.reshape(1,1,1,train_batchsz)
	confmat=zeros([yd,yd])
	total_false_label=0
	# list(self._layer_names).index('loss')
	# knn.update()
	for it in xrange(niter):
		solver.step_forward()
		end_index=start_index + train_batchsz
		if end_index>TOTAL_NUM:
			end_index=TOTAL_NUM
			unlabeled[:end_index- start_index]=use_data[start_index:end_index]==0
			unlabeled[end_index- start_index:]=use_data[:start_index+ train_batchsz- TOTAL_NUM]==0
			start_index+= train_batchsz- TOTAL_NUM
		else:
			unlabeled=(use_data[start_index:end_index]==0)
			start_index= end_index
		labeled=logical_not(unlabeled)
		label_ratio= 0.9
		
		accu_acc=0.0
		accu_iter=0.0
		new_data_exist=0
		if it>=semi_start and semi_:
			knn.update(net.blobs['fc7'].data[labeled],net.blobs['label'].data[labeled])
			print_=0
			new_data_exist=1
			thh=0.9
			if it>3000:
				thh=0.7
			elif it>2000:
				thh=0.8
			if it%4==0:
					print_=1
			
			if knn.capacity==knn.N:
				(yp,ind,accuracy)=knn.predict(net.blobs['fc7'].data[unlabeled],
					net.blobs['fc8'].data[unlabeled],
					net.blobs['label'].data[unlabeled],thh=thh,print_=print_)
				accu_acc+=accuracy
				accu_iter+=1
			else:
				new_data_exist=0
			# if it%4==0:
			# 	print "unlabeled accuracy:" ,accu_acc/4.0
				# accu_acc=0.0
			if new_data_exist:
				
				# ycp=net.blobs['label'].data[unlabeled]
				# tmp=rnd.rand(yp.shape[0])
				# tmp=tmp<0.2
				# # print "tmp ",tmp.shape,yp.shape
				# yp[:10]=randint(yd,size=10)
				# yp=net.blobs['label'].data[unlabeled][ind]
				# net.blobs['label'].data[index]=yp
				# print "dif",(net.blobs['label'].data[unlabeled]- ycp).sum()
				
				# knn.update(net.blobs['fc7'].data[unlabeled][ind],yp)
				new_add_data=arange(train_batchsz)[unlabeled][ind]
				net.blobs['label'].data[new_add_data]=yp
				unlabeled[new_add_data]=False
				# true_label=net.blobs['label'].data[new_add_data]
				# for i in xrange(true_label.shape[0]):
				# 	confmat[true_label[i],yp[i]]+=1
				total_false_label+=yp.shape[0]

		labeled=logical_not(unlabeled)
		# print net.layers[semi_layer].use_data.shape[0],net.layers[semi_layer].use_data.shape[1],net.layers[semi_layer].use_data.shape[2]
		# net.layers[semi_layer].use_data.data[0,0,ind
		net.layers[semi_layer].use_data.data[0,0,labeled]=1.0
		net.layers[semi_layer].use_data.data[0,0,unlabeled]=0.0
		# print logical_and(unlabeled,labeled).mean(),logical_or(unlabeled,labeled).mean()
		# print unlabeled[:20]
		# net.layers[semi_layer].use_data.data.fill(1)

		if True :
			# print "start opt"
			solver.step_backward()
			solver.apply_update()
		
		solver.step_extra()
		
		train_loss[it] = solver.net.blobs['loss'].data
		# print "mean",solver.test_nets[0].layers[semi_layer - 3].blobs[0].data.mean()
		# print "trainnet mean",net.layers[semi_layer].blobs[0].mean()
		# print 'Iteration', it,
		if it % test_interval == 0 :
			correct = 0.0
			test_iters=50
			for test_it in range(test_iters):
				solver.test_nets[0].forward()
				correct+=solver.test_nets[0].blobs['accuracy'].data
				# correct += sum(solver.test_nets[0].blobs['fc8'].data.argmax(1)
				# 				 == solver.test_nets[0].blobs['label'].data)
			test_acc[it // test_interval] = float(correct) / (test_iters )
			print 'Iteration', it, 'testing accuracy:',float(correct) / (test_iters)
			# filt_hist.append(solver.test_nets[0].params['conv1'][0].data)
		# print "final accuracy ",accu_acc/accu_iter
	# # plot_=1
	# # if plot_:	
	# confmat=confmat/(confmat.sum(1)[:,newaxis])
	print "total_false_label",total_false_label
	# print "confmat,",confmat
	f=open("alexnet_cifar100_result.txt","ab")
	f.write(str('-'*80)+"\n")
	f.write("iteration "+str(niter)+"\n")
	f.write("train accuracy "+str(test_acc)+"\n")
	f.write("loss "+str(train_loss[::50])+"\n \n")
	f.close()
	# f=open("alexnet_cifar100_confmat.txt","ab")
	# f.write(str('-'*80)+"\n")
	# for i in xrange(confmat.shape[0]):
	# 	f.write(confmat[i])
	# 	f.write('\n')
	# f.close()
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
	
	pass